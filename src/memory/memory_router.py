from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory.buffer_memory import BufferMemory
from src.memory.redis_memory import RedisMemory
from src.memory.episodic_memory import EpisodicMemory
from src.memory.semantic_memory import SemanticMemory


class MemoryType(str, Enum):
    BUFFER = "buffer"       # short-term: recent conversation context
    REDIS = "redis"         # long-term: user preferences / facts
    EPISODIC = "episodic"   # experience recall: past events / episodes
    SEMANTIC = "semantic"   # semantic recall: similar past interactions


ROUTER_SYSTEM_PROMPT = """You are a memory router. Given a user query, classify which memory type is most relevant.

Memory types:
- buffer: The query refers to the current conversation, recent exchanges, or "what we just talked about"
- redis: The query is about user preferences, settings, profile facts, or persistent user info
- episodic: The query asks about a specific past event, "last time", "did I ever", "remember when"
- semantic: The query seeks similar past discussions, related knowledge, or conceptually similar content

Reply with ONLY one word: buffer, redis, episodic, or semantic.
"""


class MemoryRouter:
    """
    Routes a query to the appropriate memory backend.
    Uses LLM-based intent classification.
    """

    def __init__(
        self,
        buffer: BufferMemory,
        redis: RedisMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        model: str = "gpt-4o-mini",
    ):
        self.buffer = buffer
        self.redis = redis
        self.episodic = episodic
        self.semantic = semantic
        self._llm = ChatOpenAI(model=model, temperature=0)

    def classify_intent(self, query: str) -> MemoryType:
        """Use LLM to classify the memory intent of the query."""
        response = self._llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {query}"),
        ])
        raw = response.content.strip().lower()
        try:
            return MemoryType(raw)
        except ValueError:
            return MemoryType.BUFFER  # default fallback

    def retrieve(self, query: str, top_k: int = 3) -> tuple[MemoryType, str]:
        """
        Classify query intent and retrieve relevant memory context.
        Returns (memory_type, context_text).
        """
        memory_type = self.classify_intent(query)
        context = self._fetch(memory_type, query, top_k)
        return memory_type, context

    def _fetch(self, memory_type: MemoryType, query: str, top_k: int) -> str:
        if memory_type == MemoryType.BUFFER:
            return self.buffer.get_history_text()

        if memory_type == MemoryType.REDIS:
            facts = self.redis.facts_text()
            history = self.redis.get_history_text(last_n=top_k)
            parts = []
            if facts:
                parts.append(f"User facts:\n{facts}")
            if history:
                parts.append(f"Recent history:\n{history}")
            return "\n\n".join(parts)

        if memory_type == MemoryType.EPISODIC:
            episodes = self.episodic.search_by_keyword(query, top_k=top_k)
            return self.episodic.format_episodes(episodes)

        if memory_type == MemoryType.SEMANTIC:
            return self.semantic.search_formatted(query, top_k=top_k)

        return ""
