# Báo cáo cá nhân — Day 7 · Agentic RAG

**Phạm Minh Tâm · 2A202604097 · L3B** — chạy ngày 2026-11-16. Hướng dẫn chạy lại: [src/README.md](../../src/README.md).

> **Về nhãn thủ công:** grounded answer, citation support, abstention và context grade được chấm bằng cách đọc câu trả lời và toàn văn chunk được trích, theo tiêu chí ở mục "Agent và đánh giá". Mỗi nhãn có lý do ghi trong `results/agent/<split>/<config>/reviews.json`. Mọi số liệu khác lấy từ lần chạy thật lưu trong `results/`.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** dùng lại nguyên corpus Day 6: *Clinical Nursing Skills* (OpenStax, Rice University), giấy phép **CC BY-NC-SA 4.0**, 54 trang nội dung (Chương 1 mục 1.1–1.3 và Chương 2 mục 2.1–2.4; manifest, SHA-256 và cách tính trang nằm ở repo Day 6). Dùng **index B của Day 6**: chunk theo đoạn, tối đa 800 ký tự, **330 chunk**. Chunk được xuất từ pgvector ra `data/chunks.json` (`python -m src.export_chunks`); dấu vân tay index (SHA-256 của các `content_hash`) là `aec22f207d8cc236`. Không tạo index mới, nên không có chi phí index.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (container Day 6 `pgvector/pgvector:0.8.7-pg16`), bảng `day6_chunks`, `strategy='B'`, cosine exact search (không dùng ANN). Embedding `sentence-transformers/all-MiniLM-L6-v2`, **384 chiều**, chuẩn hóa L2, chạy local trên CPU. Query dùng cùng encoder với document. Code kết nối là bản sao **không sửa** của Day 6 ([src/day6/](../../src/day6/)); `DenseRetriever` kiểm tra `model_name` của index trùng với model query. Thư viện: sentence-transformers 6.3.0, torch 2.15.0+cpu, psycopg 3.3.8, pgvector 0.5.1, openai 3.51.0. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, $0). Tải 3 model mất 37,5 s; thời gian này không tính vào độ trễ truy vấn.
- **Agent model, cách chạy và phân chia dev/test:** OpenAI `gpt-4o-mini`, `temperature=0`, tool calling bắt buộc (`tool_choice="required"`, mỗi lượt 1 tool). Chạy bằng `python -m src.run_agent --split <dev|test> --config <name>`.

  | Split | Có đáp án | Thiếu evidence | Nguồn câu hỏi |
  |---|---:|---:|---|
  | dev | 24 | 3 | q01–q24: toàn bộ 24 câu Day 6 (Day 6 đã dùng 6 câu q19–q24 làm test, nên ở Day 7 chuyển hết sang dev) · u01–u03 viết mới |
  | test | 8 | 3 | t01–t08 và u04–u06 **viết mới ngày 2026-11-16** trước mọi lần chạy Day 7, từ 8 trang chưa câu Day 6 nào dùng (p3, 15, 23, 28, 34, 38, 47, 54), phủ đủ 7 mục và 2 chương |

  Lý do: test của Day 6 đã được xem kết quả, nên không còn độc lập. Theo [RULES.md](../../RULES.md), mình tạo test mới. Câu test được diễn đạt lại, không chép nguyên câu evidence. Có 1 câu test lấy từ bảng (t05, Bảng 2.1) và 1 câu có filter người dùng (t04, `section=1.3`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* `gold` = `{source_id, page, evidence}`; câu q19 và t08 có evidence thay thế. `build_benchmark.py` kiểm tra mọi evidence (sau chuẩn hóa NFKC, dấu nháy/gạch ASCII, gộp khoảng trắng, chữ thường) **nằm trọn trong 1 chunk đúng trang**. Kết quả: 34/34 evidence đạt; nếu không đạt, build dừng. **Hit** = chunk đúng nguồn, đúng trang và chứa trọn evidence (cùng định nghĩa Day 6).
  - *Câu thiếu evidence:* chọn câu gần chủ đề (nhịp thở, tỷ lệ y tá/bệnh nhân ICU, số y tá đi Scutari, giờ học liên tục để gia hạn giấy phép, mức phạt của luật chống bạo lực nơi làm việc, SBAR). Mỗi câu được kiểm tra bằng **tìm từ khóa/regex trên toàn văn 54 trang**, không dựa vào top-k, rồi đọc các đoạn khớp. Từ khóa và kết quả ghi trong `absence_check` của [day7_new_questions.json](../../data/benchmark/day7_new_questions.json). Ví dụ: "Scutari" chỉ có ở p30, câu đó nói "a team of nurses" và không nêu số lượng. Những câu này không nằm trong mẫu số của retrieval metrics.
  - *Không đưa gold vào agent:* agent chỉ nhận `question` và `filter` (`load_cases` bỏ mọi trường khác). Gold chỉ được đọc **sau** khi agent chạy xong, để tính Context Evidence Rate.

## Retrieval và thử nghiệm

**Các phương pháp** ([src/retrievers.py](../../src/retrievers.py)). Tất cả trả về cùng dạng `{id, text, source_id, source, page, metadata, score, rank, method}`. Filter do người dùng cung cấp (`chapter`/`section`) được áp dụng ở mọi phương pháp: bằng SQL với dense, lọc danh sách chunk trước khi chấm điểm với BM25. Khi có lỗi, hệ thống báo lỗi; không tự chuyển sang phương pháp khác và không trả kết quả giả.

1. **Dense (baseline):** vector store Day 6, tương đương strategy C của Day 6 (B + filter).
2. **BM25:** Okapi BM25 (k1=1,5, b=0,75), tách từ tiếng Anh `[a-z0-9]+` sau chuẩn hóa, có danh sách stop-word nhỏ.
3. **Hybrid RRF:** gộp top-20 của dense và top-20 của BM25 bằng Reciprocal Rank Fusion (k=60). Chỉ dùng thứ hạng, không cộng hai thang điểm khác nhau.
4. **Rerank:** cross-encoder chấm lại 20 candidate của hybrid (hoặc của dense).

**Retrieval không dùng LLM, câu hỏi gốc, top-3**, context = top-3 đóng gói dưới budget. Mọi phương pháp đều xuất top-10 để tính Hit@10 và MRR. Lệnh: `python -m src.eval_retrieval --split dev`; kết quả ở `results/retrieval/<split>[-tag]/`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | dense pgvector, k=3, filter người dùng, không rerank, 2600 ký tự | Day 6 strategy C | dev | 19/24 (79%) | 23/24 (96%) | 23/24 (96%) | 35 ms/query, $0 |
| Phương pháp 2 | BM25, k=3, filter, 2600 | Câu hỏi có tên riêng/thuật ngữ (NCSBN, ANA, SMART) nên khớp từ khóa sẽ tốt | dev | 20/24 (83%) | **24/24 (100%)** | 24/24 (100%) | 0,5 ms, $0 |
| Phương pháp 3 | Hybrid RRF(dense, BM25), 20+20 candidate, k=3, filter, 2600 | Kết hợp ưu điểm hai phía | dev | 19/24 (79%) | 23/24 (96%) | 23/24 (96%) | 38 ms, $0 |
| Cải thiện 1 (phương pháp 4) | Rerank cross-encoder trên 20 candidate hybrid, k=3, filter, 2600 | Lỗi dev Hit@1 (q03, q17, q20 đúng chunk nhưng hạng 2–3) → reranker xếp chunk trả lời trực tiếp lên đầu | dev | **21/24 (88%)** | 23/24 (96%) | 23/24 (96%) | 384 ms, $0 |
| (phụ) Rerank trên dense | Rerank 20 candidate dense | Tách tác động của reranker khỏi BM25 | dev | 21/24 (88%) | 23/24 (96%) | 23/24 (96%) | 332 ms, $0 |
| Cải thiện 2 | dense, k=3, filter, **+ chunk lân cận cùng trang**, budget 4800 | Lỗi q02: câu dẫn ở c003, danh sách ở c004 → thêm chunk lân cận | dev | 19/24 (79%) | 23/24 (96%) | **24/24 (100%)** | 36 ms, $0; context TB 3783 so với 2057 ký tự |
| (đối chứng) Không filter | như trên, bỏ filter | Đo tác dụng của filter | dev | dense 19/24 · hybrid 18/24 | dense **22/24** (thêm trượt q17) · BM25 24/24 | dense 22/24 | — |
| Cấu hình đã chọn | dense + lân cận, budget 4800 (cấu hình agent `neighbors`) | Chốt trên dev | test | 5/8 (62%) | 8/8 (100%) | 8/8 (100%) | 38 ms, context TB 3870 ký tự |
| Baseline | dense, budget 2600 | | test | 5/8 (62%) | 8/8 (100%) | 8/8 (100%) | 35 ms, context TB 1957 ký tự |

Context Evidence Rate ở bảng trên là của pipeline retrieval cố định (top-3). Context Evidence Rate của **agent** (context cuối thực sự gửi model, có thể gộp nhiều lượt search) nằm trong bảng agent bên dưới. Đơn vị budget là ký tự của context đã đóng gói, gồm cả header `[id | source | page | section]`; ước lượng token ≈ ký tự/4 (2600 ký tự ≈ 650 token).

**Nhận xét retrieval:**
- BM25 tốt nhất về Evidence Hit@3 (24/24), vì câu hỏi tiếng Anh dùng đúng thuật ngữ của sách.
- Hybrid RRF **không** hơn từng thành phần. So với dense, hybrid đúng thêm q17, q20 ở top-1 nhưng mất q04, q09; so với BM25, hybrid mất q03 vì RRF kéo các chunk dense xếp cao nhưng sai lên trên.
- Rerank tăng Hit@1 từ 19 lên 21 (sửa q03, q17, q20 nhưng làm hỏng q09), nhưng chậm hơn khoảng 10 lần.
- q02 vẫn trượt ở mọi phương pháp dựa trên dense.
- Filter chỉ giúp q17 (giống Day 6).

**Hai lỗi dev, nguyên nhân và cách thử:**
1. **q02 (lỗi retrieval/chunking, ở dense, hybrid, rerank, multitool):** chunk `p005:c003` luôn được tìm thấy, nhưng nó kết thúc bằng câu dẫn *"The six steps of the NCJMM consist of the following:"*; danh sách 6 bước (có "4. Hypothesis generation") nằm ở chunk kế tiếp `p005:c004` và không bao giờ vào top-3. Agent search lại 3 lần, vẫn chỉ nhận lại c003 hoặc c000/c002, rồi **từ chối nhầm**. Giả thuyết: thêm chunk lân cận cùng trang. Kết quả: agent `neighbors` trả lời đúng "hypothesis generation" bằng `[p005:c004]`; Context Evidence của agent tăng từ 23/24 lên 24/24.
2. **bm25-q02 (citation hợp lệ nhưng không hỗ trợ) và lỗi định dạng citation:** với BM25, agent trả lời *"interpreting"*, tức một bước của mô hình **Tanner**, gán sang mô hình NCSBN, và trích `[p005:c006]` (đoạn về Tanner). ID có trong context nên qua được kiểm tra tự động, nhưng nội dung sai. Đây đúng là trường hợp "ID hợp lệ chưa chứng minh nguồn hỗ trợ". Ngoài ra, mỗi cấu hình có 4–9 lần model ghi ID vào `citations` mà quên chèn `[id]` vào câu trả lời; code từ chối và cho sửa 1 lần, nên tốn thêm 1 lần gọi LLM. Giả thuyết: prompt v2 có ví dụ định dạng và yêu cầu chunk phải nói đúng đối tượng được hỏi. Kết quả: `neighbors_v2` có **0** lần citation bị từ chối (so với 4 của `neighbors`) và trả lời đúng q02. Tuy vậy, prompt dài hơn nên chi phí trung vị cao hơn ($0.000332 so với $0.000302 mỗi câu).

**Lý do chọn cấu hình.** Tiêu chí được ghi trong [src/configs.py](../../src/configs.py) trước lần chạy agent đầu tiên:
1. Điểm grounded trung bình (/2) trên 24 câu dev cao nhất.
2. Phải từ chối đúng ít nhất 2/3 câu thiếu evidence.
3. Nếu hai cấu hình chênh nhau dưới 0,05 ở tiêu chí 1, xét lần lượt: Context Evidence Rate, số câu từ chối nhầm, chi phí trung vị, độ trễ trung vị.

`neighbors` và `neighbors_v2` cùng đạt 2,00 và 3/3; Context Evidence (24/24), từ chối nhầm (0) đều bằng nhau; `neighbors` rẻ hơn, nên **chọn `neighbors`**. Quyết định được chốt vào `results/agent/CHOSEN.json` lúc 2026-11-16T10:13:36+07:00, kèm hash code, **trước** lần chạy test đầu tiên. `run_agent.py` từ chối chạy test với cấu hình chưa chốt. Ghi chú: nếu tiêu chí ưu tiên độ ổn định định dạng hoặc độ trễ thì `neighbors_v2` tốt hơn; mình giữ đúng tiêu chí đã công bố.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi** ([src/agent.py](../../src/agent.py)):
  - *Tool:* `search(query)` (retriever theo cấu hình), `keyword_search(query)` (BM25, chỉ ở `multitool_k5`), `answer(answer, citations)`, `abstain(reason)`. Model phải gọi tool ở mọi lượt nên đầu ra luôn có cấu trúc.
  - *Điều kiện dừng:* model gọi `answer` hợp lệ hoặc `abstain`; hoặc đạt `max_searches` (3, riêng multitool là 4), khi đó model chỉ được trả lời hoặc từ chối, mọi lệnh search thêm bị từ chối và ghi vào trace; cuối cùng có giới hạn `max_searches+4` bước.
  - *Context:* cộng dồn qua các lượt search, khử trùng ID, giới hạn theo budget ký tự (2600 hoặc 4800). Chunk không vừa budget bị bỏ và ghi `dropped_by_budget`. Model chỉ thấy các chunk đã được nhận vào context; chunk đã có thì chỉ báo "Already in context". Cấu hình `neighbors` thêm chunk liền trước và liền sau cùng trang nếu còn budget.
  - *Citation:* mọi ID phải nằm trong context **và** xuất hiện dạng `[id]` trong câu trả lời. Nếu sai, model được sửa 1 lần; sai lần nữa thì từ chối.
  - *Lỗi:* lỗi retriever được ghi vào event và báo cho model là "Search failed", không thay bằng kết quả giả. Lỗi provider (ví dụ API key hết hạn ở lần chạy thử đầu tiên) làm câu đó có `status: error`, được báo riêng và không tính vào chất lượng. Nội dung tài liệu được coi là dữ liệu (quy tắc 2 trong system prompt).
- **Tiêu chí chấm** (đọc câu trả lời và toàn văn chunk được trích):
  - *Grounded:* **2** = đúng, đủ ý chính và mọi khẳng định có trong chunk được trích; **1** = đúng nhưng thiếu ý, hoặc thêm khẳng định chunk không hỗ trợ; **0** = sai, không căn cứ, hoặc từ chối câu có đáp án.
  - *Citation:* đúng nếu chunk thực sự hỗ trợ ý mà nó được gắn vào.
  - *Abstention:* đúng nếu agent đã search rồi mới từ chối và không thêm khẳng định nào.

**Dev (27 câu, mọi cấu hình; nhãn đầy đủ):**

| Cấu hình / split | Tool use | Số lượt gọi (search: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm | Context evidence (agent) | Chi phí / độ trễ trung vị |
|---|---|---|---|---|---|---|---|---|---|
| baseline_dense / dev | 27/27 | 1:25, 2:1, 3:1 | 1/1/22 · **1,88** | 26/26 | 23/23 | 3/3 | 1/24 | 23/24 | $0.0080 · 3,0 s |
| bm25 / dev | 27/27 | 1:25, 2:2 | 1/1/22 · 1,88 | 26/27 | 24/24 | 3/3 | 0/24 | 23/24 | $0.0082 · 2,6 s |
| hybrid / dev | 27/27 | 1:25, 2:1, 3:1 | 1/2/21 · 1,83 | 28/28 | 23/23 | 3/3 | 1/24 | 23/24 | $0.0077 · 2,7 s |
| rerank_hybrid / dev | 27/27 | 1:26, 2:1 | 1/1/22 · 1,88 | 27/27 | 23/23 | 3/3 | 1/24 | 23/24 | $0.0080 · 3,2 s |
| multitool_k5 / dev | 27/27 | 1:25, 3:1, 4:1 (keyword_search 6 lần) | 1/0/23 · 1,92 | 28/28 | 23/23 | 3/3 | 1/24 | 23/24 | $0.0104 · 3,4 s |
| **neighbors / dev** | 27/27 | 1:26, 2:1 | 0/0/24 · **2,00** | 30/30 | 24/24 | 3/3 | 0/24 | **24/24** | $0.0089 · 2,8 s |
| neighbors_v2 / dev | 27/27 | 1:26, 3:1 | 0/0/24 · 2,00 | 27/27 | 24/24 | 3/3 | 0/24 | 24/24 | $0.0093 · 2,5 s |
| **neighbors / test** | 11/11 | 1:10, 3:1 | 0/1/7 · **1,88** | 10/11 | 8/8 | 3/3 | 0/8 | 8/8 | $0.0040 · 2,9 s |
| **baseline_dense / test** | 11/11 | 1:10, 3:1 | 0/1/7 · **1,88** | 11/11 | 8/8 | 3/3 | 0/8 | 8/8 | $0.0034 · 3,0 s |

Bảng sinh từ `python -m src.report_tables --split <dev|test>` (dùng hàm `aggregate` của `starter/agent_metrics.py`). Script starter chạy thẳng trên nhãn test cũng cho cùng kết quả (`results/agent/test/*/starter_metrics.json`). Chi phí là tổng của cả split, tính theo giá `gpt-4o-mini` $0,15/1M token input, $0,60/1M token output; số token được lưu trong trace để tính lại khi cần.

**Nhãn 1 và 0 trên dev** (mọi câu còn lại đạt 2): q02 = 0 ở 5 cấu hình (4 cấu hình từ chối nhầm; bm25 trả lời sai với citation không hỗ trợ); q20 = 1 ở baseline, bm25, hybrid, rerank_hybrid (thêm "necessity for continuous assessment by a nurse", điều p049:c000 không nói); q04 = 1 ở hybrid (thêm "respiratory distress").

**Context grade test** (đọc cả 8 câu có đáp án, cho baseline và cấu hình chọn): cả hai cấu hình **8 câu mức 2, 0 câu mức 1, 0 câu mức 0**. Chunk gold luôn có trong context, kèm nguồn và trang; với t04, cả nhánh "not met" lẫn "achieved" đều có trong cùng chunk. Lý do từng câu ở trường `context_reason` trong `reviews.json`.

**Ví dụ trên test và dev:**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t03 (test, neighbors) | Có đáp án | `search("nurses validate data collected during assessment")` → p023:c002 (hạng 1), p022:c004, p023:c004; 7 chunk vào context (có lân cận) | "Nurses validate … by comparing and cross-referencing information … from multiple sources, such as the patient, family members, healthcare team, and medical records … [p023:c002]" | **2**; citation đúng. Khớp nguyên văn đoạn "Validate Data". |
| u06 (test, neighbors) | Thiếu evidence | `search("SBAR communication tool")`, `search("SBAR meaning")`, `search("SBAR communication tool definition")` → p036, p006, p045… không chunk nào có "SBAR"; chạm giới hạn 3 lượt | `abstain`: "The retrieved chunks do not provide the specific meanings of each letter in the SBAR communication tool." | **Từ chối đúng**. Toàn văn 54 trang không có "SBAR"; agent đổi query 3 lần rồi mới dừng, không đoán. |
| t01 (test, neighbors) | Có đáp án (lỗi) | `search("patient's ability to restore psychological balance")` → p003:c008 hạng 1 | "… called a 'coping strategy.' This involves actions or thought processes used to meet stressful situations or modify reactions to them [p003:c008]" | **1**. Tên đúng, nhưng câu 2 là định nghĩa bên ngoài, p003:c008 không có. Citation hợp lệ vẫn che được khẳng định không căn cứ. |
| q02 (dev, baseline) | Có đáp án (lỗi) | 3 lượt `search` → luôn p005:c003 (câu dẫn), không bao giờ có c004; `search_limit` | `abstain` | **0**, từ chối nhầm. Lỗi do chunking cắt danh sách; `neighbors` sửa được. |
| q02 (dev, bm25) | Có đáp án (lỗi) | 2 lượt `search` → p005:c003, c006 (Tanner), p006 | "… is called 'interpreting' … [p005:c006]" | **0**; citation **không hỗ trợ**. Lấy bước của mô hình khác. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace agent: `results/agent/<split>/<config>/traces.json`. Mỗi câu lưu từng lần gọi LLM (token, ms, chi phí), từng search (tool, query, filter, top-k, kết quả kèm rank/score/trang/mục, chunk nhận vào, chunk bị bỏ do budget), context cuối (`context`, `context_ids`), câu trả lời, citation, lỗi và cấu hình.
  - Nhãn: `reviews.json` cùng thư mục. Tự động: `summary_auto.json`. Bảng: `results/agent/<split>/agent_table.md`. Trace retrieval: `results/retrieval/<split>[-tag]/traces/<method>.json`.
  - Chi phí cả thí nghiệm (9 lượt chạy agent, 211 câu): 353.545 token input + 28.527 token output = **$0.068**. Retrieval và embedding chạy local nên $0.
  - Độ trễ: embedding cộng search dense khoảng 35 ms; rerank khoảng 330–380 ms; một câu agent khoảng 2,5–3,4 s (trung vị), chủ yếu là thời gian gọi LLM.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - *Trước/sau trên dev:* `neighbors` sửa đúng lỗi mục tiêu (q02, từ 0 lên 2) và gián tiếp sửa q20 (từ 1 lên 2), nâng grounded từ 1,88 lên 2,00 và Context Evidence của agent từ 23/24 lên 24/24. Đổi lại, context tăng khoảng 1,9 lần (3844 so với 2053 ký tự), token input tăng 22% và chi phí dev tăng 11%. Rerank và multitool cải thiện Hit@1 ở mức retrieval, nhưng **không** cải thiện câu trả lời của agent, vì agent đọc cả top-3; thứ hạng 1 hay 3 không quan trọng bằng việc evidence có nằm trong context hay không.
  - *Test (sau khi chốt):* `neighbors` **không hơn** baseline: cả hai đạt 1,88, 3/3 abstention, 0 từ chối nhầm. `neighbors` có thêm 1 citation đặt sai chỗ (t08 `[p054:c006]` gắn vào ý delegation, trong khi delegation nằm ở c005) và tốn hơn khoảng 18%. Lý do: 8 câu test không có trường hợp danh sách bị tách khỏi câu dẫn như q02, nên lợi ích của chunk lân cận không thể hiện; ngược lại, context dài hơn cho model thêm chỗ để thêm thắt. Ở cả hai cấu hình, lỗi duy nhất là **khẳng định thêm không căn cứ** (t01, t08), không phải lỗi retrieval.
  - *Giới hạn:*
    - Mẫu nhỏ: dev 24 câu, test 8 câu. Chênh 1 câu ở test bằng 0,25 điểm /2, nên không kết luận được cấu hình nào tốt hơn ngoài lỗi q02 cụ thể.
    - Dev gồm cả 24 câu Day 6, mà Day 6 đã tối ưu chunking trên chúng, nên các con số dev cao hơn mức thực tế.
    - Một tài liệu, một ngôn ngữ (tiếng Anh). BM25 mạnh một phần vì câu hỏi dùng đúng thuật ngữ của sách.
    - `temperature=0` nhưng OpenAI không đảm bảo kết quả tất định; trong trace, cùng một câu có diễn đạt khác nhau giữa các cấu hình.
    - Nhãn do một người chấm và không chấm mù (người chấm biết cấu hình); ranh giới giữa mức 1 và 2 ("thêm khẳng định không căn cứ") có tính chủ quan, ví dụ câu thứ hai của t06 ở baseline được chấm 2 sau khi đối chiếu p054:c003.
    - Câu "thiếu evidence" chỉ được kiểm tra trong corpus 54 trang, không phải toàn bộ sách 1347 trang.
    - Kiểm tra citation tự động chỉ xét ID; việc nguồn có hỗ trợ hay không phải đọc thủ công.
  - *Nếu làm tiếp:* sửa từ gốc ở bước chunking (giữ câu dẫn cùng danh sách, gắn tiêu đề mục vào chunk) thay vì vá bằng chunk lân cận. Thêm bước tự kiểm tra từng câu với chunk được trích, hoặc giới hạn câu trả lời chỉ gồm khẳng định trích được, để giảm lỗi kiểu t01/t08. Tăng số câu test có danh sách hoặc bảng.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.

**Công cụ AI:** Claude Code (Anthropic) hỗ trợ viết code (retriever, agent, script đánh giá), soạn nháp câu hỏi benchmark mới từ văn bản trang sách, chạy thí nghiệm, **chấm nháp các nhãn thủ công** theo tiêu chí ở trên (có lý do cho từng nhãn) và viết nháp báo cáo. Cách kiểm tra:
- Mọi số liệu lấy từ lần chạy thật trong `results/`.
- Evidence được script kiểm tra 34/34.
- Câu thiếu evidence được kiểm tra bằng tìm kiếm trên toàn văn.
- Kết quả dense khớp với Day 6 (dev Hit@3 23/24 = 17/18 + 6/6 của Day 6 C).
- Test mock của starter vẫn pass.
- Học viên đọc lại các nhãn và lý do trong `reviews.json`.
