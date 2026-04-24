"""
Metrics for benchmark evaluation.
- Response Relevance: does the response address the query?
- Context Utilization: does the response use provided memory context?
- Token Efficiency: useful tokens / total tokens ratio
"""

import re
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


RELEVANCE_PROMPT = """Rate how well the AI response addresses the user query.
Score: 0-10 (10 = perfectly relevant, 0 = completely irrelevant).
Reply with ONLY a number between 0 and 10."""

CONTEXT_UTILIZATION_PROMPT = """Given the memory context and the AI response, rate how well the response 
utilizes the provided memory context (0-10).
10 = response clearly draws from and references the memory context.
0 = response ignores the memory context entirely.
Reply with ONLY a number between 0 and 10."""


class BenchmarkMetrics:
    """Computes benchmark metrics for a single turn."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._llm = ChatOpenAI(model=model, temperature=0)
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def response_relevance(self, query: str, response: str) -> float:
        """LLM-judged relevance score 0-10."""
        result = self._llm.invoke([
            SystemMessage(content=RELEVANCE_PROMPT),
            HumanMessage(content=f"Query: {query}\n\nAI Response: {response}"),
        ])
        return self._parse_score(result.content)

    def context_utilization(self, context: str, response: str) -> float:
        """LLM-judged context utilization score 0-10. Returns 0 if no context."""
        if not context or not context.strip():
            return 0.0
        result = self._llm.invoke([
            SystemMessage(content=CONTEXT_UTILIZATION_PROMPT),
            HumanMessage(content=f"Memory context:\n{context}\n\nAI Response: {response}"),
        ])
        return self._parse_score(result.content)

    def token_efficiency(self, prompt_tokens: int, response_tokens: int) -> float:
        """
        Ratio: response_tokens / (prompt_tokens + response_tokens).
        Higher = more of the budget went to output vs context overhead.
        """
        total = prompt_tokens + response_tokens
        if total == 0:
            return 0.0
        return round(response_tokens / total, 4)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text)) if text else 0

    def _parse_score(self, text: str) -> float:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", text.strip())
        if match:
            return min(10.0, float(match.group(1)))
        return 0.0
