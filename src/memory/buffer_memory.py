from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage


class BufferMemory:
    """Short-term in-memory conversation buffer."""

    def __init__(self, session_id: str = "default", max_messages: int = 20):
        self.session_id = session_id
        self.max_messages = max_messages
        self._history = InMemoryChatMessageHistory()

    def add_interaction(self, human: str, ai: str) -> None:
        self._history.add_user_message(human)
        self._history.add_ai_message(ai)
        self._trim()

    def get_messages(self) -> list:
        return self._history.messages

    def get_history_text(self) -> str:
        lines = []
        for m in self.get_messages():
            role = "Human" if isinstance(m, HumanMessage) else "AI"
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    def _trim(self) -> None:
        msgs = self._history.messages
        if len(msgs) > self.max_messages:
            self._history.messages = msgs[-self.max_messages:]

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history.messages)
