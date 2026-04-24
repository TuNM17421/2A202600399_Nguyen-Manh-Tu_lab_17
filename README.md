# 2A202600399_Nguyen-Manh-Tu_lab_17

> **Lab 17 — Build Multi-Memory Agent với LangGraph**  
> Học viên: Nguyễn Mạnh Tú — MSSV: 2A202600399

Agent hội thoại tích hợp **4 memory backends** (short-term, long-term, episodic, semantic) điều phối bởi **LangGraph StateGraph**, với conflict resolution tự động và context window management.

---

## Kiến trúc tổng quan

### LangGraph Flow

```
User Input
    │
    ▼
┌─────────────────┐
│ classify_intent │  ← LLM phân loại: buffer / redis / episodic / semantic
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ retrieve_memory │  ← Lấy context từ cả 4 backends
└────────┬────────┘
         │
    ▼
┌──────────────────────┐
│  generate_response   │  ← Build prompt với 4 section riêng, gọi LLM
└────────┬─────────────┘
         │
    ▼
┌──────────────────┐
│  update_profile  │  ← Extract facts, detect & resolve conflicts (Redis)
└────────┬─────────┘
         │
    ▼
┌─────────────────┐
│ persist_memory  │  ← Ghi vào Buffer + Redis + Episodic + Chroma
└────────┬────────┘
         │
    ▼
  Response
```

### 4 Memory Backends

| Loại | Backend | Vai trò | TTL |
|------|---------|---------|-----|
| **Short-term** | `InMemoryChatMessageHistory` | Lưu N turn gần nhất, sliding window | Process lifetime |
| **Long-term profile** | Redis (Docker) | User facts/preferences, structured key-value | 24h (configurable) |
| **Episodic** | JSON append-log | Lưu từng interaction kèm tags & timestamp | Không tự xóa |
| **Semantic** | Chroma + OpenAI embeddings | Tìm kiếm theo semantic similarity | Không tự xóa |

### MemoryState (LangGraph TypedDict)

```python
class MemoryState(TypedDict):
    user_input: str
    session_id: str
    memory_type: str            # router output
    user_profile: dict          # Redis facts
    recent_conversation: str    # Buffer
    episodes: list[dict]        # Episodic log
    semantic_hits: list[str]    # Chroma results
    memory_budget: int          # token budget remaining
    profile_updates: dict       # facts extracted this turn
    conflicts_resolved: list    # ConflictLog entries
    messages: list
    response: str
```

---

## Cài đặt & chạy

### Yêu cầu
- Python 3.11+
- Docker

### Setup

```bash
# 1. Start Redis
docker-compose up -d

# 2. Cấu hình environment
copy .env.example .env
# Mở .env, điền OPENAI_API_KEY

# 3. Cài dependencies
python -m pip install -r requirements.txt
```

### Chạy

```bash
# Interactive chat
python main.py

# Benchmark (12 conversations, so sánh with/without memory)
python benchmark/run_benchmark.py

# Unit tests (conflict handling)
python -m pytest tests/ -v
```

---

## Giải thích các thành phần chính

### MemoryRouter (`src/memory/memory_router.py`)
Dùng LLM (`gpt-4o-mini`, temperature=0) để classify intent của query thành 1 trong 4 loại memory. Backend được classify là **primary** sẽ dùng `top_k=5`, các backend còn lại dùng `top_k=2`.

### ConflictHandler (`src/memory/conflict_handler.py`)
Extract profile facts từ user utterance bằng LLM, sau đó resolve conflict theo rule **"giá trị mới luôn thắng"**. Conflict được log vào episodic memory với tag `["conflict", "profile_update"]`.

```
User: "Tôi dị ứng sữa bò."        → profile: {allergy: "sữa bò"}
User: "Nhầm, tôi dị ứng đậu nành." → profile: {allergy: "đậu nành"}  ✅
```

### ContextWindowManager (`src/context/context_manager.py`)
Giới hạn token budget (mặc định 3000 tokens) với 4-level eviction priority:

```
Evict first                                    Evict last
    │                                               │
  buffer → episodic → redis → semantic (highest)
```

---

## Kết quả benchmark

> Chi tiết đầy đủ: [BENCHMARK.md](BENCHMARK.md)

| Metric | With Memory | Without Memory | Delta |
|--------|-------------|----------------|-------|
| Response Relevance (avg/10) | 8.92 | 8.42 | +0.50 |
| Context Utilization (avg/10) | 5.92 | 0.00 | +5.92 |
| Token Efficiency | 0.3589 | 0.7977 | -0.44 |
| Memory Hit Rate | 58.0% | N/A | — |

12 conversations (12 scenarios), bao phủ: profile recall, conflict update, episodic recall, semantic retrieval, trim/token budget.

---

## Reflection

> Chi tiết đầy đủ: [REFLECTION.md](REFLECTION.md)

- **Memory rủi ro nhất:** Redis — lưu PII plaintext, không mã hóa, không auth mặc định
- **Deletion:** `RedisMemory.clear()` + `EpisodicMemory.clear_session()` + `SemanticMemory.clear()` theo `session_id`
- **TTL:** Redis 24h ✅ | Chroma & Episodic không có TTL ⚠️
- **Limitation chính:** Router chỉ pick 1 backend per turn; LLM fact extraction có thể sai với câu phủ định; Episodic JSON không thread-safe

---

## Bonus đạt được

| Bonus | Status |
|-------|--------|
| Redis thật (Docker) chạy ổn | ✅ |
| Chroma thật (local persistent) chạy ổn | ✅ |
| Token counting bằng `tiktoken` | ✅ |
| LangGraph flow rõ ràng, dễ explain | ✅ |
| LLM-based fact extraction có error handling | ✅ |

