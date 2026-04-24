import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory.buffer_memory import BufferMemory
from src.memory.redis_memory import RedisMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory
from src.memory.memory_router import MemoryRouter, MemoryType
from src.context.context_manager import ContextWindowManager
from src.graph.memory_graph import MemoryGraphBuilder
from src.graph.memory_state import MemoryState


class MemoryAgent:
    """
    LLM agent integrating all 4 memory backends via a LangGraph StateGraph.

    Graph flow:
        classify_intent → retrieve_memory → generate_response → persist_memory → END

    MemoryState carries: user_profile (Redis), recent_conversation (Buffer),
    episodes (Episodic), semantic_hits (Chroma), memory_budget (token count).

    When use_memory=False the agent bypasses the graph and calls the LLM directly
    (used for no-memory baseline in benchmarks).
    """

    def __init__(
        self,
        session_id: str,
        model: str = None,
        max_context_tokens: int = None,
        use_memory: bool = True,
    ):
        self.session_id = session_id
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.use_memory = use_memory
        self._max_tokens = max_context_tokens or int(os.getenv("CONTEXT_MAX_TOKENS", 3000))

        # Memory backends (shared between graph and direct access)
        self.buffer = BufferMemory(session_id=session_id)
        self.redis = RedisMemory(session_id=session_id)
        self.episodic = EpisodicMemory(session_id=session_id)
        self.semantic = SemanticMemory(session_id=session_id)

        # LangGraph — compiled once, reused per chat() call
        self._graph_builder = MemoryGraphBuilder(
            buffer=self.buffer,
            redis=self.redis,
            episodic=self.episodic,
            semantic=self.semantic,
            model=self.model,
            max_tokens=self._max_tokens,
        )
        self._graph = self._graph_builder.compile()

        # Kept for backward-compat (benchmark, main.py)
        self.router = MemoryRouter(
            buffer=self.buffer,
            redis=self.redis,
            episodic=self.episodic,
            semantic=self.semantic,
            model=self.model,
        )
        self.ctx_manager = ContextWindowManager(
            max_tokens=self._max_tokens,
            model=self.model,
        )

        self._last_state: MemoryState | None = None
        self._llm = ChatOpenAI(model=self.model, temperature=0.7)

    # ── Public API ─────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Run user_input through the LangGraph (with memory) or direct LLM (no memory).
        Returns the AI response string.
        """
        if self.use_memory:
            return self._chat_with_graph(user_input)
        return self._chat_direct(user_input)

    def store_user_fact(self, key: str, value: str) -> None:
        """Explicitly upsert a fact in Redis (overwrites existing value for key)."""
        self.redis.store_fact(key, value)

    # ── Graph path ─────────────────────────────────────────────────────────

    def _chat_with_graph(self, user_input: str) -> str:
        initial_state: MemoryState = {
            "user_input": user_input,
            "session_id": self.session_id,
            "memory_type": "buffer",
            "user_profile": {},
            "recent_conversation": "",
            "episodes": [],
            "semantic_hits": [],
            "memory_budget": self._max_tokens,
            "profile_updates": {},
            "conflicts_resolved": [],
            "messages": [],
            "response": "",
        }
        final_state = self._graph.invoke(initial_state)
        self._last_state = final_state

        # Sync ctx_manager slot for backward-compat (benchmark token summary)
        mem_type = final_state.get("memory_type", "buffer")
        self.ctx_manager.set_slot("buffer", final_state.get("recent_conversation", ""))
        self.ctx_manager.set_slot("redis", str(final_state.get("user_profile", {})))
        self.ctx_manager.set_slot("episodic", str(final_state.get("episodes", [])))
        self.ctx_manager.set_slot("semantic", "\n".join(final_state.get("semantic_hits", [])))

        return final_state["response"]

    # ── No-memory baseline path ────────────────────────────────────────────

    def _chat_direct(self, user_input: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Be concise and helpful."),
            HumanMessage(content=user_input),
        ]
        result = self._llm.invoke(messages)
        return result.content

    # ── Properties for benchmark / main.py ────────────────────────────────

    @property
    def last_memory_type(self) -> str:
        if self._last_state:
            return self._last_state.get("memory_type", "none")
        return "none"

    @property
    def last_token_summary(self) -> str:
        return self.ctx_manager.summary()

    @property
    def last_conflicts(self) -> list[dict]:
        """Return conflict resolutions from the last chat() call."""
        if self._last_state:
            return self._last_state.get("conflicts_resolved", [])
        return []
