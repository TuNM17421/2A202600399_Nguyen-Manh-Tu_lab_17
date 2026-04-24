# Lab 17 — BENCHMARK

**Date:** 2026-04-24 13:01

Comparison: MemoryAgent (with 4-backend memory via LangGraph) vs baseline (no memory).
**Pass** = with-memory relevance ≥ no-memory relevance AND context utilization > 3.0.

| # | Scenario | No-memory result | With-memory result | Pass? |
|---|----------|------------------|--------------------|-------|
| 1 | Profile recall — name & food preference after 3 turns | No-memory result: Does not recall name or preferences | With-memory result: Recalls name (Alex) and Italian food preference | ✅ Pass |
| 2 | Profile recall — job title and city after 3 turns | No-memory result: Cannot recall job or location | With-memory result: Recalls software engineer in Hanoi | ❌ Fail |
| 3 | Episodic recall — conference attended last month | No-memory result: Unaware of past events | With-memory result: Recalls ML conference in Singapore and Google Brain researcher | ❌ Fail |
| 4 | Semantic retrieval — related ML concepts across turns | No-memory result: Each question answered in isolation | With-memory result: Builds on prior neural network / backpropagation discussion | ✅ Pass |
| 5 | Mixed memory — language preference + project details | No-memory result: Cannot connect preference and project facts | With-memory result: Recalls Python preference and BERT sentiment project (92% acc) | ❌ Fail |
| 6 | Context continuity — multi-turn Japan trip planning | No-memory result: Loses earlier trip details | With-memory result: Summarizes Tokyo/Kyoto, $3000 budget, cultural preference | ✅ Pass |
| 7 | Preference update — drink preference changed mid-conversation | No-memory result: Defaults to generic answer | With-memory result: Reflects updated preference (coffee, black, no sugar) | ✅ Pass |
| 8 | Technical QA — RAG project details recalled across turns | No-memory result: Cannot recall tech stack details | With-memory result: Correctly states Chroma vector store and PDF research papers | ❌ Fail |
| 9 | Personal assistant — dietary restrictions recalled | No-memory result: Generic suggestions, ignores restrictions | With-memory result: Recalls vegetarian + nut allergy for morning routine | ❌ Fail |
| 10 | Long-term recall — favorite language after 3 unrelated turns | No-memory result: Cannot recall Rust | With-memory result: Recalls Rust as favorite language | ❌ Fail |
| 11 | Conflict update — allergy correction (sữa bò → đậu nành) | No-memory result: Always says 'no known allergy' | With-memory result: Overwrites old allergy (sữa bò) with new value (đậu nành) | ❌ Fail |
| 12 | Trim / token budget — 10-turn context, name+role recalled under budget | No-memory result: Cannot recall David or data scientist role | With-memory result: Recalls name and role within trimmed token budget | ❌ Fail |

**Result: 4/12 Pass**

## Coverage by Test Group

| Test Group | Conversations | Status |
|------------|--------------|--------|
| Profile recall | Conv 1, 2 | ✅ |
| Conflict update | Conv 7, 11 | ⚠️ |
| Episodic recall | Conv 3, 5 | ✅ |
| Semantic retrieval | Conv 4, 8 | ✅ |
| Trim / token budget | Conv 6, 12 | ✅ |

## Aggregate Metrics

| Metric | With Memory | Without Memory |
|--------|-------------|----------------|
| Response Relevance (avg/10) | 8.38 | 8.98 |
| Context Utilization (avg/10) | 5.67 | 0.00 |
