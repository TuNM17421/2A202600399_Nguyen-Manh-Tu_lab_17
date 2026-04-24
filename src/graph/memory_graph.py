"""
LangGraph graph for the Multi-Memory Agent.

Graph flow:
    classify_intent
         ↓
    retrieve_memory
         ↓
    generate_response
         ↓
    persist_memory
         ↓
       END
"""

import os
import tiktoken
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.memory_state import MemoryState
from src.memory.buffer_memory import BufferMemory
from src.memory.redis_memory import RedisMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory
from src.memory.memory_router import MemoryRouter, MemoryType
from src.memory.conflict_handler import ConflictHandler


AGENT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to multiple memory systems.
Use the provided memory context to give accurate, personalized, and contextually relevant responses.
If memory context is present, reference relevant facts naturally.
Be concise and helpful."""


def _build_prompt_from_state(state: MemoryState) -> list:
    """
    Build the LLM message list with 4 clearly separated memory sections.
    Each section maps to a specific memory backend.
    """
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

    context_parts = []

    # Section 1: Long-term user profile (Redis)
    if state.get("user_profile"):
        profile_lines = "\n".join(f"  - {k}: {v}" for k, v in state["user_profile"].items())
        context_parts.append(f"[USER PROFILE — long-term facts]\n{profile_lines}")

    # Section 2: Recent conversation (BufferMemory)
    if state.get("recent_conversation"):
        context_parts.append(f"[RECENT CONVERSATION — short-term]\n{state['recent_conversation']}")

    # Section 3: Episodic memory (JSON log)
    if state.get("episodes"):
        ep_lines = []
        for ep in state["episodes"]:
            ep_lines.append(f"  [{ep.get('timestamp', '')[:10]}] Human: {ep.get('human', '')}")
            ep_lines.append(f"  [{ep.get('timestamp', '')[:10]}] AI: {ep.get('ai', '')}")
        context_parts.append(f"[EPISODIC MEMORY — past events]\n" + "\n".join(ep_lines))

    # Section 4: Semantic hits (Chroma)
    if state.get("semantic_hits"):
        sem_lines = "\n".join(f"  {h}" for h in state["semantic_hits"])
        context_parts.append(f"[SEMANTIC MEMORY — similar past interactions]\n{sem_lines}")

    if context_parts:
        messages.append(SystemMessage(content="\n\n".join(context_parts)))

    messages.append(HumanMessage(content=state["user_input"]))
    return messages


class MemoryGraphBuilder:
    """
    Builds and compiles a LangGraph StateGraph for the multi-memory agent.
    All node functions close over the memory backends.
    """

    def __init__(
        self,
        buffer: BufferMemory,
        redis: RedisMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        model: str = "gpt-4o-mini",
        max_tokens: int = 3000,
    ):
        self.buffer = buffer
        self.redis = redis
        self.episodic = episodic
        self.semantic = semantic
        self.model = model
        self.max_tokens = max_tokens
        self._llm = ChatOpenAI(model=model, temperature=0.7)
        self._router = MemoryRouter(
            buffer=buffer, redis=redis, episodic=episodic,
            semantic=semantic, model=model,
        )
        self._conflict_handler = ConflictHandler(model=model)
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ── Node 1: classify_intent ────────────────────────────────────────────

    def classify_intent(self, state: MemoryState) -> dict:
        """Classify query intent → pick which memory backend to prioritise."""
        memory_type = self._router.classify_intent(state["user_input"])
        return {"memory_type": memory_type.value}

    # ── Node 2: retrieve_memory ────────────────────────────────────────────

    def retrieve_memory(self, state: MemoryState) -> dict:
        """
        Retrieve from all 4 backends.
        The memory_type from classify_intent determines retrieval depth:
        - Prioritised backend gets top_k=5, others get top_k=2.
        """
        query = state["user_input"]
        primary = state.get("memory_type", "buffer")
        top_k_primary = 5
        top_k_secondary = 2

        # Long-term user profile (Redis)
        user_profile = self.redis.get_all_facts()

        # Recent conversation (Buffer)
        recent_conversation = self.buffer.get_history_text()

        # Episodic memory (JSON log)
        top_k_ep = top_k_primary if primary == "episodic" else top_k_secondary
        episodes = self.episodic.search_by_keyword(query, top_k=top_k_ep)

        # Semantic memory (Chroma)
        top_k_sem = top_k_primary if primary == "semantic" else top_k_secondary
        sem_results = self.semantic.search(query, top_k=top_k_sem)
        semantic_hits = [r["content"] for r in sem_results]

        # Token budget calculation
        used = (
            len(self._enc.encode(str(user_profile)))
            + len(self._enc.encode(recent_conversation))
            + len(self._enc.encode(str(episodes)))
            + sum(len(self._enc.encode(h)) for h in semantic_hits)
        )
        budget = max(0, self.max_tokens - used)

        return {
            "user_profile": user_profile,
            "recent_conversation": recent_conversation,
            "episodes": episodes,
            "semantic_hits": semantic_hits,
            "memory_budget": budget,
        }

    # ── Node 3: generate_response ──────────────────────────────────────────

    def generate_response(self, state: MemoryState) -> dict:
        """Build prompt from 4 memory sections and call LLM."""
        messages = _build_prompt_from_state(state)
        ai_message = self._llm.invoke(messages)
        return {
            "messages": messages,
            "response": ai_message.content,
        }

    # ── Node 4: update_profile ────────────────────────────────────────────

    def update_profile(self, state: MemoryState) -> dict:
        """
        Extract facts from user_input, detect conflicts with existing profile,
        resolve by keeping the newest value, and persist to Redis.
        """
        new_facts = self._conflict_handler.extract_facts(state["user_input"])
        if not new_facts:
            return {"profile_updates": {}, "conflicts_resolved": []}

        existing_profile = state.get("user_profile") or {}
        merged_profile, conflicts = self._conflict_handler.resolve(existing_profile, new_facts)

        # Persist each updated fact to Redis (HSET overwrites existing key)
        for key, value in new_facts.items():
            self.redis.store_fact(key, value)

        # Log conflicts as episodic entries so they're auditable
        conflict_dicts = [c.to_dict() for c in conflicts]
        for c in conflicts:
            self.episodic.log_episode(
                human=f"[CONFLICT] User corrected '{c.key}': '{c.old_value}' → '{c.new_value}'",
                ai="[System: profile updated]",
                tags=["conflict", "profile_update", c.key],
            )

        return {
            "user_profile": merged_profile,
            "profile_updates": new_facts,
            "conflicts_resolved": conflict_dicts,
        }

    # ── Node 5: persist_memory ─────────────────────────────────────────────

    def persist_memory(self, state: MemoryState) -> dict:
        """Write the interaction to all 4 backends."""
        human = state["user_input"]
        ai = state["response"]
        self.buffer.add_interaction(human, ai)
        self.redis.add_interaction(human, ai)
        self.episodic.log_episode(human, ai, tags=self._infer_tags(human))
        self.semantic.add_interaction(human, ai)
        return {}  # no state mutation needed

    # ── Graph compilation ──────────────────────────────────────────────────

    def compile(self):
        """Compile and return the LangGraph StateGraph."""
        graph = StateGraph(MemoryState)

        graph.add_node("classify_intent", self.classify_intent)
        graph.add_node("retrieve_memory", self.retrieve_memory)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("update_profile", self.update_profile)
        graph.add_node("persist_memory", self.persist_memory)

        graph.set_entry_point("classify_intent")
        graph.add_edge("classify_intent", "retrieve_memory")
        graph.add_edge("retrieve_memory", "generate_response")
        graph.add_edge("generate_response", "update_profile")
        graph.add_edge("update_profile", "persist_memory")
        graph.add_edge("persist_memory", END)

        return graph.compile()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _infer_tags(self, text: str) -> list[str]:
        tags = []
        text_lower = text.lower()
        keyword_map = {
            "preference": ["like", "prefer", "favorite", "love", "hate", "dislike"],
            "fact": ["my name", "i am", "i work", "i live", "i have"],
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "task": ["help me", "can you", "please", "create", "write", "make"],
        }
        for tag, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)
        return tags
