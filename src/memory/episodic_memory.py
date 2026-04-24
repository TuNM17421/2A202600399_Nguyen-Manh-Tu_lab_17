import json
import os
import uuid
from datetime import datetime
from pathlib import Path


class EpisodicMemory:
    """
    JSON-based episodic memory log.
    Each episode = one interaction with metadata tags.
    """

    def __init__(self, session_id: str, log_path: str = None):
        self.session_id = session_id
        log_path = log_path or os.getenv("EPISODIC_LOG_PATH", "./data/episodic_log.json")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("[]", encoding="utf-8")

    # --- Write ---

    def log_episode(self, human: str, ai: str, tags: list[str] = None) -> str:
        episode_id = str(uuid.uuid4())
        episode = {
            "id": episode_id,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "human": human,
            "ai": ai,
            "tags": tags or [],
        }
        episodes = self._load()
        episodes.append(episode)
        self._save(episodes)
        return episode_id

    # --- Read ---

    def get_all(self, session_only: bool = True) -> list[dict]:
        episodes = self._load()
        if session_only:
            return [e for e in episodes if e["session_id"] == self.session_id]
        return episodes

    def search_by_tag(self, tag: str) -> list[dict]:
        return [e for e in self.get_all() if tag in e.get("tags", [])]

    def search_by_keyword(self, keyword: str, top_k: int = 5) -> list[dict]:
        kw = keyword.lower()
        matches = [
            e for e in self.get_all()
            if kw in e["human"].lower() or kw in e["ai"].lower()
        ]
        return matches[-top_k:]

    def get_recent(self, n: int = 5) -> list[dict]:
        return self.get_all()[-n:]

    def format_episodes(self, episodes: list[dict]) -> str:
        if not episodes:
            return ""
        lines = []
        for e in episodes:
            lines.append(f"[{e['timestamp'][:10]}] Human: {e['human']}")
            lines.append(f"[{e['timestamp'][:10]}] AI: {e['ai']}")
        return "\n".join(lines)

    # --- Internal ---

    def _load(self) -> list[dict]:
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, episodes: list[dict]) -> None:
        self.log_path.write_text(
            json.dumps(episodes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_session(self) -> None:
        episodes = [e for e in self._load() if e["session_id"] != self.session_id]
        self._save(episodes)
