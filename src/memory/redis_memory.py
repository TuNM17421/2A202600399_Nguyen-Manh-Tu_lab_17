import json
import os
import redis
from datetime import datetime


class RedisMemory:
    """Long-term persistent memory backed by Redis."""

    def __init__(self, session_id: str, redis_url: str = None, ttl: int = None):
        self.session_id = session_id
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.ttl = ttl or int(os.getenv("REDIS_TTL", 86400))
        self._client = redis.from_url(self.redis_url, decode_responses=True)
        self._history_key = f"memory:history:{session_id}"
        self._facts_key = f"memory:facts:{session_id}"

    # --- Conversation history ---

    def add_interaction(self, human: str, ai: str) -> None:
        entry = json.dumps({
            "human": human,
            "ai": ai,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self._client.rpush(self._history_key, entry)
        self._client.expire(self._history_key, self.ttl)

    def get_history(self, last_n: int = 20) -> list[dict]:
        raw = self._client.lrange(self._history_key, -last_n, -1)
        return [json.loads(r) for r in raw]

    def get_history_text(self, last_n: int = 10) -> str:
        entries = self.get_history(last_n)
        lines = []
        for e in entries:
            lines.append(f"Human: {e['human']}")
            lines.append(f"AI: {e['ai']}")
        return "\n".join(lines)

    # --- User facts / preferences ---

    def store_fact(self, key: str, value: str) -> None:
        self._client.hset(self._facts_key, key, value)
        self._client.expire(self._facts_key, self.ttl)

    def get_fact(self, key: str) -> str | None:
        return self._client.hget(self._facts_key, key)

    def get_all_facts(self) -> dict:
        return self._client.hgetall(self._facts_key)

    def facts_text(self) -> str:
        facts = self.get_all_facts()
        if not facts:
            return ""
        return "\n".join(f"- {k}: {v}" for k, v in facts.items())

    # --- Misc ---

    def clear(self) -> None:
        self._client.delete(self._history_key, self._facts_key)

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except redis.exceptions.ConnectionError:
            return False
