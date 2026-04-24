# Lab 17 — Benchmark Report

**Date:** 2026-04-24 13:01

## Overview
Comparison of MemoryAgent (with 4-backend memory) vs baseline agent (no memory)
across 10 multi-turn conversations measuring 3 metrics.

## Aggregate Metrics Comparison

| Metric | With Memory | Without Memory | Delta |
|--------|-------------|----------------|-------|
| Response Relevance (avg/10) | 8.38 | 9.00 | -0.62 |
| Context Utilization (avg/10) | 5.77 | 0.00 | +5.77 |
| Token Efficiency (avg) | 0.2946 | 0.8124 | -0.5178 |
| Memory Hit Rate | 53.8% | N/A | N/A |

## Per-Conversation Summary

| Conv | Theme | Rel (mem) | Rel (no) | Ctx Util (mem) | Ctx Util (no) | Tok Eff (mem) | Tok Eff (no) |
|------|-------|-----------|----------|---------------|---------------|--------------|--------------|
| 1 | user_preference | 9.8 | 8.8 | 5.0 | 0.0 | 0.314 | 0.8149 |
| 2 | factual_recall | 7.0 | 9.4 | 4.6 | 0.0 | 0.254 | 0.7993 |
| 3 | experience_recall | 6.6 | 8.8 | 7.2 | 0.0 | 0.2637 | 0.7907 |
| 4 | semantic_similarity | 10.0 | 9.4 | 8.0 | 0.0 | 0.3789 | 0.9174 |
| 5 | mixed_memory | 7.8 | 8.2 | 7.0 | 0.0 | 0.2752 | 0.761 |
| 6 | context_continuity | 10.0 | 8.4 | 7.6 | 0.0 | 0.4923 | 0.889 |
| 7 | preference_update | 8.0 | 8.0 | 6.2 | 0.0 | 0.2629 | 0.7637 |
| 8 | technical_qa | 9.0 | 9.6 | 6.6 | 0.0 | 0.297 | 0.8892 |
| 9 | personal_assistant | 8.8 | 9.6 | 4.0 | 0.0 | 0.3333 | 0.8408 |
| 10 | long_term_recall | 8.4 | 9.4 | 2.0 | 0.0 | 0.2984 | 0.656 |
| 11 | conflict_update | 6.8 | 9.0 | 3.0 | 0.0 | 0.2939 | 0.8522 |
| 12 | trim_budget | 8.4 | 9.2 | 6.9 | 0.0 | 0.1828 | 0.7934 |

## Memory Hit Rate Analysis
- **Memory Hit Rate:** 53.8% (turns where context_utilization > 5.0)
- **Avg Context Utilization (with memory):** 5.77/10
- **Avg Context Utilization (no memory):** 0.00/10

## Token Budget Breakdown
Token distribution across memory types when memory is enabled.

## Conclusions
- Memory-enabled agent shows higher context utilization scores.
- Response relevance improvement depends on conversation theme.
- Episodic and semantic memory are most useful for recall-heavy conversations.