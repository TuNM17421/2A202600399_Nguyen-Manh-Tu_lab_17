"""
Tests for ConflictHandler — explicit conflict resolution logic.

Run:
    python -m pytest tests/test_conflict_handling.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from src.memory.conflict_handler import ConflictHandler, ConflictLog


# ── Unit tests (no LLM calls) ──────────────────────────────────────────────

class TestConflictResolve:
    """Test resolve() logic without hitting the LLM."""

    def setup_method(self):
        # Patch LLM so no API calls needed
        with patch("src.memory.conflict_handler.ChatOpenAI"):
            self.handler = ConflictHandler()

    def test_no_conflict_adds_new_fact(self):
        existing = {"name": "Alex"}
        new_facts = {"allergy": "milk"}
        merged, conflicts = self.handler.resolve(existing, new_facts)
        assert merged["allergy"] == "milk"
        assert conflicts == []

    def test_conflict_new_value_wins(self):
        """Core rubric test: user corrects allergy from milk → soy."""
        existing = {"allergy": "milk"}
        new_facts = {"allergy": "soy"}
        merged, conflicts = self.handler.resolve(existing, new_facts)
        assert merged["allergy"] == "soy", "New value must override old value"
        assert len(conflicts) == 1
        assert conflicts[0].key == "allergy"
        assert conflicts[0].old_value == "milk"
        assert conflicts[0].new_value == "soy"

    def test_conflict_log_to_dict(self):
        log = ConflictLog("allergy", "milk", "soy")
        d = log.to_dict()
        assert d == {"key": "allergy", "old_value": "milk", "new_value": "soy"}

    def test_multiple_facts_partial_conflict(self):
        existing = {"name": "Alex", "allergy": "milk", "city": "Hanoi"}
        new_facts = {"allergy": "soy", "job": "engineer"}
        merged, conflicts = self.handler.resolve(existing, new_facts)
        assert merged["allergy"] == "soy"
        assert merged["name"] == "Alex"       # unchanged
        assert merged["city"] == "Hanoi"      # unchanged
        assert merged["job"] == "engineer"    # new fact added
        assert len(conflicts) == 1
        assert conflicts[0].key == "allergy"

    def test_same_value_no_conflict(self):
        """Updating with the same value should NOT produce a conflict."""
        existing = {"allergy": "milk"}
        new_facts = {"allergy": "milk"}
        merged, conflicts = self.handler.resolve(existing, new_facts)
        assert merged["allergy"] == "milk"
        assert conflicts == []

    def test_empty_new_facts(self):
        existing = {"name": "Alex"}
        merged, conflicts = self.handler.resolve(existing, {})
        assert merged == existing
        assert conflicts == []


# ── Integration test for extract_facts (mocked LLM) ───────────────────────

class TestExtractFacts:
    """Test extract_facts() with mocked LLM response."""

    def _make_handler_with_response(self, llm_reply: str) -> ConflictHandler:
        mock_msg = MagicMock()
        mock_msg.content = llm_reply
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_msg
        with patch("src.memory.conflict_handler.ChatOpenAI", return_value=mock_llm):
            handler = ConflictHandler()
        handler._llm = mock_llm
        return handler

    def test_extract_allergy_fact(self):
        handler = self._make_handler_with_response('{"allergy": "milk"}')
        facts = handler.extract_facts("I'm allergic to milk.")
        assert facts == {"allergy": "milk"}

    def test_extract_corrected_allergy(self):
        handler = self._make_handler_with_response('{"allergy": "soy"}')
        facts = handler.extract_facts("Actually I'm allergic to soy, not milk.")
        assert facts == {"allergy": "soy"}

    def test_extract_no_facts(self):
        handler = self._make_handler_with_response("{}")
        facts = handler.extract_facts("What's the capital of France?")
        assert facts == {}

    def test_extract_invalid_json_returns_empty(self):
        handler = self._make_handler_with_response("Sorry, I can't parse that.")
        facts = handler.extract_facts("Some text")
        assert facts == {}


# ── Rubric scenario test ───────────────────────────────────────────────────

class TestAllergyScenario:
    """
    Rubric-required scenario:
        User: Tôi dị ứng sữa bò.
        User: À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
        Expected profile: allergy = đậu nành
    """

    def test_allergy_conflict_resolution(self):
        with patch("src.memory.conflict_handler.ChatOpenAI"):
            handler = ConflictHandler()

        # Turn 1: user states allergy = sữa bò
        profile = {}
        profile, conflicts1 = handler.resolve(profile, {"allergy": "sữa bò"})
        assert profile["allergy"] == "sữa bò"
        assert conflicts1 == []

        # Turn 2: user corrects to đậu nành
        profile, conflicts2 = handler.resolve(profile, {"allergy": "đậu nành"})
        assert profile["allergy"] == "đậu nành", (
            "After correction, allergy must be 'đậu nành', not 'sữa bò'"
        )
        assert len(conflicts2) == 1
        assert conflicts2[0].old_value == "sữa bò"
        assert conflicts2[0].new_value == "đậu nành"
