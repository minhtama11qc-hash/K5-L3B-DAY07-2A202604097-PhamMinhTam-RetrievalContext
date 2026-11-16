"""Local web UI to try the agent (standard library HTTP server, no extra dependencies).

    python -m src.app            # then open http://127.0.0.1:8765

Needs the same setup as run_agent: Day 6 pgvector container running and OPENAI_API_KEY in .env.
Every question costs a few hundredths of a cent (gpt-4o-mini). Runs are not saved to results/.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer

from .agent import Agent
from .common import CHUNKS_FILE, load_env, read_json
from .configs import CONFIGS
from .eval_retrieval import build

HOST, PORT = "127.0.0.1", 8765
STATE: dict = {}

EXAMPLES = [
    ("What are the four adaptive modes in Roy's adaptation model?", ""),
    ("In the NCSBN clinical judgment model, which step is about coming up with possible explanations for the patient's problem?", ""),
    ("Why should basic care for a critically unstable patient not be delegated?", ""),
    ("What do beneficence and nonmaleficence require?", "1.2"),
    ("What does each letter of the SBAR communication tool stand for?", ""),
    ("How many nurses did Florence Nightingale bring with her to Scutari?", ""),
]


def setup() -> None:
    load_env()
    chunks = read_json(CHUNKS_FILE)
    STATE["by_id"] = {c["id"]: c for c in chunks}
    t0 = time.perf_counter()
    STATE["retrievers"] = build(["dense", "bm25", "hybrid", "rerank_hybrid"], chunks)
    STATE["agents"] = {}
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")


def agent_for(name: str) -> Agent:
    if name not in STATE["agents"]:
        STATE["agents"][name] = Agent(CONFIGS[name], STATE["retrievers"], STATE["by_id"])
    return STATE["agents"][name]


def ask(payload: dict) -> dict:
    question = str(payload.get("question", "")).strip()
    if not question:
        return {"error": "Câu hỏi trống."}
    name = payload.get("config") or "neighbors"
    if name not in CONFIGS:
        return {"error": f"Không có cấu hình {name}."}
    filters = {}
    if payload.get("chapter"):
        filters["chapter"] = int(payload["chapter"])
    if payload.get("section"):
        filters["section"] = str(payload["section"]).strip()
    try:
        r = agent_for(name).run(question, filters or None)
    except Exception as exc:  # show provider / store errors as they are
        return {"error": f"{type(exc).__name__}: {exc}"}
    by_id = STATE["by_id"]
    r["cited"] = [{"id": i, "page": by_id[i]["page"], "section": by_id[i]["metadata"].get("section"),
                   "text": by_id[i]["text"]} for i in r["citations"]]
    r["context_chunks"] = [{"id": i, "page": by_id[i]["page"], "section": by_id[i]["metadata"].get("section"),
                            "text": by_id[i]["text"]} for i in r["context_ids"]]
    r.pop("context", None)
    return r


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            page = PAGE.replace("__CONFIGS__", json.dumps(
                [{"name": n, "note": c.note, **{k: v for k, v in asdict(c).items() if k in ("method", "top_k", "budget", "max_searches", "neighbors", "keyword_tool", "prompt")}}
                 for n, c in CONFIGS.items()])).replace("__EXAMPLES__", json.dumps(EXAMPLES))
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            return self._send(404, b"not found", "text/plain")
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error":"bad json"}', "application/json")
        result = ask(payload)
        self._send(200, json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def log_message(self, fmt, *args) -> None:  # keep the console quiet
        pass


PAGE = r"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nursing RAG Agent</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--muted:#667085;--line:#e3e6ea;--accent:#2f6fde;--ok:#16794c;--okbg:#e7f6ee;--warn:#9a5b00;--warnbg:#fff4e0;--err:#b42318;--errbg:#fdecea;--code:#f1f3f6}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b1f27;--fg:#e6e8ec;--muted:#98a2b3;--line:#2b313c;--accent:#6b9bff;--ok:#5fd39a;--okbg:#14301f;--warnbg:#3a2a0e;--warn:#f5b84d;--errbg:#3b1512;--err:#ff8a7a;--code:#232833}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
textarea{width:100%;min-height:70px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit;resize:vertical}
.row{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;align-items:end}
label{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:4px}
select,input{padding:7px 9px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);font:inherit}
input{width:90px}button{padding:9px 18px;border:0;border-radius:8px;background:var(--accent);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.ex{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.ex span{font-size:12.5px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;cursor:pointer;color:var(--muted)}.ex span:hover{color:var(--fg);border-color:var(--accent)}
.note{font-size:12.5px;color:var(--muted);margin-top:8px}
.badge{display:inline-block;font-size:12px;font-weight:600;padding:2px 10px;border-radius:999px}
.answered{background:var(--okbg);color:var(--ok)}.abstain{background:var(--warnbg);color:var(--warn)}.error{background:var(--errbg);color:var(--err)}
.answer{font-size:16px;margin:10px 0 0;white-space:pre-wrap}.cite{color:var(--accent);font-weight:600;cursor:pointer}
.stats{display:flex;flex-wrap:wrap;gap:18px;color:var(--muted);font-size:13px;margin-top:12px}.stats b{color:var(--fg)}
h2{font-size:15px;margin:0 0 10px}.chunk{border-left:3px solid var(--line);padding:6px 10px;margin:8px 0;background:var(--code);border-radius:0 6px 6px 0;font-size:13.5px;white-space:pre-wrap}
.chunk.cited{border-left-color:var(--accent)}.chunk .h{font-size:12px;color:var(--muted);margin-bottom:4px;font-family:ui-monospace,Consolas,monospace}
.step{font-size:13.5px;padding:6px 0;border-bottom:1px dashed var(--line)}.step:last-child{border:0}.step code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12.5px}
details summary{cursor:pointer;color:var(--muted);font-size:13.5px}.hidden{display:none}
</style></head><body><main>
<h1>Nursing RAG Agent</h1>
<p class="sub">Hỏi về sách <i>Clinical Nursing Skills</i> (OpenStax, chương 1–2, 54 trang). Agent tự tìm bằng chứng trong pgvector Day 6, rồi trả lời kèm nguồn hoặc từ chối.</p>
<div class="card">
  <textarea id="q" placeholder="Nhập câu hỏi bằng tiếng Anh, ví dụ: What is compassion fatigue?"></textarea>
  <div class="ex" id="ex"></div>
  <div class="row">
    <label>Cấu hình<select id="cfg"></select></label>
    <label>Chương (filter)<input id="chapter" placeholder="1 / 2"></label>
    <label>Mục (filter)<input id="section" placeholder="vd 2.3"></label>
    <button id="go">Hỏi agent</button>
  </div>
  <div class="note" id="cfgnote"></div>
</div>
<div id="out" class="hidden">
  <div class="card"><span id="badge" class="badge"></span><p class="answer" id="answer"></p><p class="note" id="reason"></p><div class="stats" id="stats"></div></div>
  <div class="card" id="citedcard"><h2>Nguồn được trích</h2><div id="cited"></div></div>
  <div class="card"><h2>Các bước của agent</h2><div id="steps"></div></div>
  <div class="card"><details><summary id="ctxsum">Context cuối gửi cho model</summary><div id="ctx"></div></details></div>
</div>
</main>
<script>
const CONFIGS=__CONFIGS__, EXAMPLES=__EXAMPLES__;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const short=id=>id.split(":").slice(2).join(":");
CONFIGS.forEach(c=>{const o=document.createElement("option");o.value=c.name;o.textContent=c.name+(c.name==="neighbors"?" (đã chọn)":c.name==="baseline_dense"?" (baseline)":"");$("cfg").append(o)});
$("cfg").value="neighbors";
const showNote=()=>{const c=CONFIGS.find(x=>x.name===$("cfg").value);$("cfgnote").textContent=`${c.note} · method=${c.method}, top_k=${c.top_k}, budget=${c.budget} ký tự, tối đa ${c.max_searches} lượt search${c.neighbors?", có chunk lân cận":""}${c.keyword_tool?", có keyword_search":""}, prompt ${c.prompt}`};
$("cfg").onchange=showNote;showNote();
EXAMPLES.forEach(([q,s])=>{const e=document.createElement("span");e.textContent=q.length>60?q.slice(0,58)+"…":q;e.title=q;e.onclick=()=>{$("q").value=q;$("section").value=s;$("chapter").value=""};$("ex").append(e)});
$("q").addEventListener("keydown",e=>{if(e.key==="Enter"&&(e.ctrlKey||e.metaKey))ask()});
$("go").onclick=ask;
async function ask(){
  const q=$("q").value.trim();if(!q)return;
  $("go").disabled=true;$("go").textContent="Đang chạy…";
  try{
    const r=await (await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,config:$("cfg").value,chapter:$("chapter").value.trim(),section:$("section").value.trim()})})).json();
    render(r);
  }catch(e){render({error:String(e)})}
  $("go").disabled=false;$("go").textContent="Hỏi agent";
}
function render(r){
  $("out").classList.remove("hidden");
  const st=r.error?"error":r.status;
  $("badge").className="badge "+st;$("badge").textContent={answered:"Đã trả lời",abstain:"Từ chối: không đủ bằng chứng",error:"Lỗi"}[st]||st;
  if(r.error){$("answer").textContent=r.error;["reason","stats","cited","steps","ctx"].forEach(i=>$(i).innerHTML="");return}
  $("answer").innerHTML=esc(r.answer).replace(/\[([^\]]+)\]/g,(m,id)=>`<span class="cite" onclick="document.getElementById('c-${CSS.escape(id)}')?.scrollIntoView({behavior:'smooth',block:'center'})">[${esc(short(id))}]</span>`);
  $("reason").textContent=r.reason?("Lý do: "+r.reason):"";
  $("stats").innerHTML=`<span>Search: <b>${r.searches}</b></span><span>Gọi LLM: <b>${r.llm_calls}</b></span><span>Token: <b>${r.prompt_tokens}</b> in / <b>${r.completion_tokens}</b> out</span><span>Chi phí: <b>$${r.cost_usd.toFixed(5)}</b></span><span>Thời gian: <b>${r.elapsed_s}s</b></span><span>Context: <b>${r.context_chars}</b> ký tự</span>`;
  $("citedcard").classList.toggle("hidden",!r.cited.length);
  $("cited").innerHTML=r.cited.map(c=>`<div class="chunk cited" id="c-${esc(c.id)}"><div class="h">${esc(short(c.id))} · trang ${c.page} · mục ${esc(c.section)}</div>${esc(c.text)}</div>`).join("");
  $("steps").innerHTML=r.events.filter(e=>e.event!=="llm_call").map(e=>{
    if(e.event==="search")return `<div class="step">🔎 <code>${esc(e.tool)}</code> “${esc(e.query)}”${e.filter?` · filter ${esc(JSON.stringify(e.filter))}`:""} → ${e.error?`<span style="color:var(--err)">${esc(e.error)}</span>`:e.results.map(x=>`<code>${esc(short(x.id))}</code> (${x.score})`).join(", ")} · nhận vào context: ${e.admitted.length}${e.dropped_by_budget.length?` · bỏ do budget: ${e.dropped_by_budget.length}`:""}</div>`;
    if(e.event==="answer")return `<div class="step">✅ Trả lời, trích ${e.citations.map(c=>`<code>${esc(short(c))}</code>`).join(", ")}</div>`;
    if(e.event==="abstain")return `<div class="step">⛔ Từ chối: ${esc(e.reason)}</div>`;
    if(e.event==="invalid_citations")return `<div class="step">⚠️ Citation bị từ chối (${esc((e.bad||[]).map(short).join(", "))}); model được sửa 1 lần</div>`;
    if(e.event==="search_limit")return `<div class="step">🛑 Hết lượt search (${e.searches})</div>`;
    return `<div class="step">${esc(e.event)} ${esc(JSON.stringify(e))}</div>`}).join("");
  $("ctxsum").textContent=`Context cuối gửi cho model (${r.context_chunks.length} chunk)`;
  $("ctx").innerHTML=r.context_chunks.map(c=>`<div class="chunk${r.citations.includes(c.id)?" cited":""}"><div class="h">${esc(short(c.id))} · trang ${c.page} · mục ${esc(c.section)}</div>${esc(c.text)}</div>`).join("");
}
</script></body></html>"""


def main() -> None:
    setup()
    print(f"UI: http://{HOST}:{PORT}  (Ctrl+C để dừng)")
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
