# Code Day 7 — Agentic RAG trên vector store Day 6

Báo cáo: [report/individual/2A202604097_PhamMinhTam.md](../report/individual/2A202604097_PhamMinhTam.md)

## Thành phần

| File | Vai trò |
|---|---|
| `day6/` | Bản sao **không sửa** `embeddings.py`, `models.py`, `store.py` của Day 6 (all-MiniLM-L6-v2, pgvector, cosine exact search) |
| `export_chunks.py` | Xuất 330 chunk của index B (Day 6) từ pgvector → `data/chunks.json` |
| `build_benchmark.py` | Ghép câu hỏi Day 6 + câu mới Day 7 → `data/benchmark/{questions,agent_cases}.json`, kiểm tra mọi evidence nằm trọn trong 1 chunk đúng trang |
| `retrievers.py` | `dense` (pgvector), `bm25`, `hybrid` (RRF k=60), `rerank` (cross-encoder ms-marco-MiniLM-L-6-v2) |
| `context.py` | Đóng gói context: khử trùng, budget ký tự, thêm chunk lân cận cùng trang |
| `eval_retrieval.py` | Benchmark retrieval không dùng LLM: Hit@1, Evidence Hit@3, Hit@10, MRR@10, Context Evidence Rate |
| `agent.py` | Agent OpenAI tool calling (`search`, `keyword_search`, `answer`, `abstain`), kiểm tra citation, trace đầy đủ |
| `configs.py` | 7 cấu hình agent + **tiêu chí chọn công bố trước** |
| `run_agent.py` | Chạy agent trên 1 split; test bị chặn nếu chưa có `results/agent/CHOSEN.json` |
| `show_trace.py` | In trace để chấm nhãn thủ công |
| `report_tables.py` | Ghép nhãn thủ công (qua `starter/agent_metrics.aggregate`) → bảng agent |
| `app.py` | Giao diện web local để hỏi thử agent: `python -m src.app` → http://127.0.0.1:8765 (không lưu vào `results/`) |

## Chạy lại

Yêu cầu: Python 3.11+ (đã chạy 3.13.7, Windows 11, CPU), Docker, container pgvector của Day 6, OpenAI API key.

```bash
# 1. Môi trường
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env                                    # điền OPENAI_API_KEY

# 2. Bật vector store Day 6 (trong repo Day 6, thư mục starter/)
docker compose -p day6_student_starter up -d --wait    # đã có dữ liệu index A/B từ Day 6

# 3. Dữ liệu và benchmark (đã có sẵn trong repo; chạy lại để kiểm tra)
python -m src.export_chunks
python -m src.build_benchmark --day6-questions <repo Day 6>/benchmark/questions.json

# 4. Retrieval trên dev (không tốn phí)
python -m src.eval_retrieval --split dev
python -m src.eval_retrieval --split dev --no-filter --tag nofilter
python -m src.eval_retrieval --split dev --methods dense --budget 4800 --neighbors --tag neighbors

# 5. Agent trên dev, mỗi cấu hình
python -m src.run_agent --split dev --config baseline_dense   # bm25, hybrid, rerank_hybrid, multitool_k5, neighbors, neighbors_v2
python -m src.show_trace --split dev --config baseline_dense  # đọc và điền results/agent/dev/<config>/reviews.json
python -m src.report_tables --split dev

# 6. Chốt cấu hình (results/agent/CHOSEN.json), rồi mới chạy test
python -m src.eval_retrieval --split test --methods dense
python -m src.run_agent --split test --config baseline_dense
python -m src.run_agent --split test --config neighbors
python -m src.report_tables --split test
python starter/agent_metrics.py --cases data/benchmark/agent_cases.json --traces results/agent/test/neighbors/traces.json --reviews results/agent/test/neighbors/reviews.json --split test --output results/agent/test/neighbors/starter_metrics.json
```

Agent dùng `temperature=0`, nhưng OpenAI không đảm bảo kết quả tất định; chạy lại có thể khác vài câu. Kết quả trong báo cáo là lần chạy được lưu ở `results/`.
