"""
ConflictHandler — extracts profile facts from user utterances and
resolves conflicts when a user updates a previously stated fact.

Rule: the most recent value for any fact key always wins.
A conflict log is kept so the system can explain what changed.
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


EXTRACT_FACTS_PROMPT = """Extract profile facts from the user's message as a JSON object.
Only extract clear, explicit personal facts (name, allergy, preference, job, location, etc.).
Keys should be short snake_case strings (e.g. "allergy", "name", "favorite_food").
If the message contains a correction ("actually", "I meant", "not X but Y"), extract the corrected value.
If no facts are present, return {}.

Reply with ONLY valid JSON, no explanation.

Examples:
User: "My name is Alex and I'm allergic to milk."
→ {"name": "Alex", "allergy": "milk"}

User: "Actually I'm allergic to soy, not milk."
→ {"allergy": "soy"}

User: "What's the capital of France?"
→ {}
"""


class ConflictLog:
    """Record of a single fact conflict resolution."""

    def __init__(self, key: str, old_value: str, new_value: str):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }

    def __repr__(self) -> str:
        return f"ConflictLog({self.key}: '{self.old_value}' → '{self.new_value}')"


class ConflictHandler:
    """
    Detects when a user fact contradicts an existing profile entry
    and resolves it by keeping the newest value.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self._llm = ChatOpenAI(model=model, temperature=0)

    def extract_facts(self, user_input: str) -> dict[str, str]:
        """Use LLM to extract key-value facts from the user message."""
        response = self._llm.invoke([
            SystemMessage(content=EXTRACT_FACTS_PROMPT),
            HumanMessage(content=user_input),
        ])
        raw = response.content.strip()
        try:
            facts = json.loads(raw)
            if isinstance(facts, dict):
                return {str(k): str(v) for k, v in facts.items()}
        except json.JSONDecodeError:
            pass
        return {}

    def resolve(
        self,
        existing_profile: dict[str, str],
        new_facts: dict[str, str],
    ) -> tuple[dict[str, str], list[ConflictLog]]:
        """
        Merge new_facts into existing_profile.
        New value always overrides old value for the same key.

        Returns:
            merged_profile: dict with conflicts resolved
            conflicts: list of ConflictLog for any overwritten facts
        """
        conflicts: list[ConflictLog] = []
        merged = dict(existing_profile)

        for key, new_val in new_facts.items():
            old_val = merged.get(key)
            if old_val is not None and old_val != new_val:
                conflicts.append(ConflictLog(key, old_val, new_val))
            merged[key] = new_val

        return merged, conflicts
