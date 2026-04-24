# Lab 17 — Reflection: Privacy, PII, TTL & Technical Limitations

**Author:** Nguyen Manh Tu — 2A202600399  
**Date:** 2026-04-24

---

## 1. Memory nào giúp agent nhất?

**Redis (long-term profile)** giúp nhất trong hầu hết conversations.  
Lý do: nó là backend duy nhất lưu trữ **structured facts** (key-value) — tên, dị ứng, sở thích — theo session_id và persist qua nhiều sessions. Khi agent cần trả lời câu hỏi về user ("Tôi dị ứng gì?", "Tôi thích màu gì?"), Redis trả về chính xác, không phụ thuộc vào context window hay embedding similarity.

**Semantic memory (Chroma)** đứng thứ hai — đặc biệt hữu ích cho các câu hỏi liên quan đến kiến thức kỹ thuật hoặc chủ đề đã được thảo luận trước đó trong conversation.

---

## 2. Memory nào rủi ro nhất nếu retrieve sai?

**Redis (long-term profile)** là rủi ro nhất khi retrieve sai, vì hai lý do:

### 2a. PII được lưu plaintext không mã hóa
Profile facts hiện tại lưu dạng raw string trong Redis hash:
```
memory:facts:{session_id} → {"allergy": "sữa bò", "name": "Alex", "city": "Hanoi"}
```
Nếu Redis instance bị compromise (không dùng AUTH, không TLS), toàn bộ PII của user bị lộ. Các loại dữ liệu nhạy cảm có thể bị lưu vô tình: tên thật, địa chỉ, tình trạng sức khỏe (dị ứng, bệnh), thông tin tài chính.

### 2b. Conflict handler dùng LLM để extract facts
`ConflictHandler.extract_facts()` gọi GPT-4o-mini để parse user utterance. Nếu LLM misclassify (e.g., "Tôi không dị ứng sữa bò nữa" → extract `{"allergy": "sữa bò"}` thay vì xóa key), profile bị lưu sai và **agent sẽ tiếp tục đưa ra lời khuyên nguy hiểm** (e.g., recommend món có sữa cho người thực sự dị ứng).

### 2c. Episodic memory lưu toàn bộ raw conversation
`episodic_log.json` append không giới hạn. Trong production, file này có thể chứa hàng nghìn entries bao gồm mọi thứ user từng nói — không có consent mechanism, không có retention policy.

---

## 3. Nếu user yêu cầu xóa memory, xóa ở đâu?

Theo thứ tự cần thực hiện khi user yêu cầu "xóa dữ liệu của tôi" (right to erasure — GDPR Article 17):

| Backend | Key cần xóa | Method hiện có |
|---------|-------------|----------------|
| **Redis** | `memory:history:{session_id}`, `memory:facts:{session_id}` | `RedisMemory.clear()` — xóa cả 2 keys |
| **Episodic JSON** | Tất cả entries có `session_id == target` | `EpisodicMemory.clear_session()` — filter và ghi lại |
| **Chroma** | Collection `lab17_semantic_{session_id}` | `SemanticMemory.clear()` — `delete_collection()` |
| **Buffer** | In-memory list | `BufferMemory.clear()` — không persist, mất khi restart |

**Vấn đề hiện tại:** Chưa có endpoint/command thống nhất để xóa tất cả backends cùng lúc theo session_id. Cần implement một `MemoryAgent.delete_all_memory()` gọi `clear()` trên cả 4 backends.

---

## 4. TTL và Consent

### TTL hiện tại
- **Redis**: TTL = 86400 giây (24h) mặc định, cấu hình qua `REDIS_TTL` env var. Sau 24h, Redis tự xóa history và facts.
- **Episodic JSON**: Không có TTL — file tích lũy vô hạn. 
- **Chroma**: Không có TTL — collection tồn tại cho đến khi `delete_collection()` được gọi. 
- **Buffer**: In-memory, tự mất khi process restart.

### Vấn đề consent
Hiện tại system **không hỏi consent** trước khi lưu PII vào Redis hay episodic log. Trong production, cần:
1. Hiển thị privacy notice khi user bắt đầu session.
2. Cho phép user opt-out khỏi long-term memory (chỉ dùng buffer).
3. Cung cấp `/delete-my-data` endpoint.

---

## 5. Technical Limitations

### 5a. Memory router chỉ chọn 1 backend per turn
`classify_intent()` trả về 1 `MemoryType`. Nếu query cần cả profile facts lẫn episodic context (e.g., "Dựa vào dị ứng của tôi và chuyến đi Singapore, tôi nên ăn gì?"), router chỉ pick 1 backend — context từ backend còn lại bị retrieve ở `top_k=2` (secondary), có thể bị miss hoặc bị trim.

### 5b. Fact extraction bằng LLM có thể sai
`ConflictHandler.extract_facts()` phụ thuộc vào GPT-4o-mini. LLM có thể:
- Extract fact từ câu phủ định: "Tôi **không** thích rau cải" → `{"preference": "rau cải"}` (sai)
- Bỏ sót fact trong câu phức tạp
- Gây extra API call mỗi turn (tăng latency và cost)

### 5c. Chroma không scale tốt với nhiều sessions
Mỗi session tạo một Chroma collection riêng (`lab17_semantic_{session_id}`). Với N users, có N collections. Chroma local mode không có index management cho số lượng lớn collections — performance degradation khi > 1000 sessions.

### 5d. Episodic JSON không phù hợp cho production
Append-only JSON file:
- Không thread-safe (concurrent writes sẽ corrupt file)
- Không có index — `search_by_keyword()` phải scan toàn bộ file O(n)
- File size tăng vô hạn nếu không có retention policy

### 5e. Token budget eviction có thể xóa context quan trọng
`ContextWindowManager` evict theo priority cố định: buffer → episodic → redis → semantic. Nếu semantic hits (highest priority) quá dài, nó vẫn bị truncate khi over budget, mất context quan trọng nhất.

### 5f. Không có authentication giữa agent và Redis
Redis hiện chạy không có password (`redis-server --appendonly yes`). Trong production, cần:
```yaml
command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
```

---

## 6. Tóm tắt rủi ro theo mức độ nghiêm trọng

| Rủi ro | Mức độ | Backend | Giải pháp đề xuất |
|--------|--------|---------|------------------|
| PII lưu plaintext trong Redis | Cao | Redis | Encrypt at rest, Redis AUTH + TLS |
| Episodic log không có TTL/deletion | Cao | JSON | Thêm retention policy, implement right-to-erasure |
| LLM extract fact sai từ câu phủ định | Trung bình | ConflictHandler | Validation prompt + user confirmation cho sensitive facts |
| Chroma không scale theo sessions | Trung bình | Chroma | Dùng shared collection với metadata filter theo session_id |
| Không có consent mechanism | Trung bình | Tất cả | Privacy notice + opt-out flag |
| Redis không có password | Trung bình | Redis | Thêm requirepass trong docker-compose |
| JSON file không thread-safe | Thấp (lab) | JSON | Thay bằng SQLite hoặc PostgreSQL trong production |
