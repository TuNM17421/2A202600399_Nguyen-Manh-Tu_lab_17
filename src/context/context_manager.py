import os
import tiktoken


# 4-level priority hierarchy for eviction (highest priority = kept last)
# Level 1 (lowest): buffer recent msgs
# Level 2: episodic context
# Level 3: redis facts
# Level 4 (highest): semantic context (most relevant)
PRIORITY_ORDER = ["buffer", "episodic", "redis", "semantic"]


class ContextWindowManager:
    """
    Manages the context window for the LLM prompt.
    - Tracks token usage across memory context slots
    - Auto-trims when approaching the limit
    - Evicts by 4-level priority when over budget
    """

    def __init__(self, max_tokens: int = None, model: str = "gpt-4o-mini"):
        self.max_tokens = max_tokens or int(os.getenv("CONTEXT_MAX_TOKENS", 3000))
        self.model = model
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

        # Slots: dict[priority_label] -> text
        self._slots: dict[str, str] = {k: "" for k in PRIORITY_ORDER}

    # --- Slot management ---

    def set_slot(self, slot: str, text: str) -> None:
        """Set a memory context slot. slot must be one of PRIORITY_ORDER."""
        if slot not in PRIORITY_ORDER:
            raise ValueError(f"Unknown slot '{slot}'. Valid: {PRIORITY_ORDER}")
        self._slots[slot] = text

    def get_context(self) -> str:
        """Build and return trimmed context string within token budget."""
        self._auto_trim()
        parts = []
        for slot in reversed(PRIORITY_ORDER):  # highest priority first
            if self._slots[slot]:
                parts.append(f"[{slot.upper()} MEMORY]\n{self._slots[slot]}")
        return "\n\n".join(parts)

    def token_usage(self) -> dict[str, int]:
        """Return token count per slot."""
        return {slot: self._count(text) for slot, text in self._slots.items()}

    def total_tokens(self) -> int:
        return sum(self.token_usage().values())

    def is_over_budget(self) -> bool:
        return self.total_tokens() > self.max_tokens

    # --- Trimming ---

    def _auto_trim(self) -> None:
        """Evict lowest-priority slots first until within budget."""
        if not self.is_over_budget():
            return

        for slot in PRIORITY_ORDER:  # evict lowest priority first
            if not self._slots[slot]:
                continue
            self._slots[slot] = self._trim_text(self._slots[slot])
            if not self.is_over_budget():
                return

        # Final hard trim: truncate lowest priority slot
        for slot in PRIORITY_ORDER:
            if self._slots[slot]:
                budget = self.max_tokens - sum(
                    self._count(t) for s, t in self._slots.items() if s != slot
                )
                self._slots[slot] = self._truncate_to_tokens(self._slots[slot], max(budget, 0))
                break

    def _trim_text(self, text: str) -> str:
        """Remove oldest half of lines from a text block."""
        lines = text.split("\n")
        if len(lines) <= 2:
            return ""
        return "\n".join(lines[len(lines) // 2:])

    def _truncate_to_tokens(self, text: str, max_tok: int) -> str:
        tokens = self._enc.encode(text)
        if len(tokens) <= max_tok:
            return text
        return self._enc.decode(tokens[:max_tok])

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text)) if text else 0

    def summary(self) -> str:
        usage = self.token_usage()
        total = sum(usage.values())
        lines = [f"Token budget: {total}/{self.max_tokens}"]
        for slot in reversed(PRIORITY_ORDER):
            lines.append(f"  [{slot}] {usage[slot]} tokens")
        return "\n".join(lines)
