from typing import TypedDict


class MemoryState(TypedDict):
    """
    LangGraph state flowing through the multi-memory agent graph.

    Nodes:
        classify_intent  → sets memory_type
        retrieve_memory  → fills user_profile, recent_conversation,
                           episodes, semantic_hits, memory_budget
        generate_response → sets response, messages
        update_profile   → extracts facts, resolves conflicts, updates Redis
        persist_memory   → writes interaction to all backends (side-effect only)
    """

    # ── Input ──────────────────────────────────────────────────────────────
    user_input: str
    session_id: str

    # ── Router output ──────────────────────────────────────────────────────
    memory_type: str          # "buffer" | "redis" | "episodic" | "semantic"

    # ── 4 Memory sections (injected into prompt separately) ────────────────
    user_profile: dict        # long-term facts from Redis (key → value)
    recent_conversation: str  # last N turns from BufferMemory
    episodes: list[dict]      # matching episodes from EpisodicMemory
    semantic_hits: list[str]  # similar past texts from SemanticMemory (Chroma)

    # ── Context management ─────────────────────────────────────────────────
    memory_budget: int        # remaining token budget after retrieval

    # ── Conflict handling ──────────────────────────────────────────────────
    profile_updates: dict     # new facts extracted from this turn's user_input
    conflicts_resolved: list  # list of ConflictLog.to_dict() for overwritten facts

    # ── LLM IO ────────────────────────────────────────────────────────────
    messages: list            # built by generate_response node
    response: str             # final AI response
