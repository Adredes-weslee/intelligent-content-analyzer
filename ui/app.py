import hashlib
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Intelligent Content Analyzer", layout="wide")

st.markdown(
    """
    <style>
    button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"],
    .stForm button[data-testid="baseButton-primary"] {
        font-size: 1.25rem !important;
        padding: 1rem 1.4rem !important;
        width: 100% !important;
        min-height: 3rem !important;
        border-radius: 0.5rem !important;
    }
    button[data-testid="baseButton-primary"] p {
        font-size: 1.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Session state defaults ----
for k, v in {
    "restrict_last_doc": True,
    "show_raw": False,
    "show_debug": False,
    "use_judge": False,
    "dense_candidates": 50,
    "doc_id": None,
    "qa_result": None,
    "summary": None,
    "docs": [],
    "doc_index": {},
}.items():
    st.session_state.setdefault(k, v)


def _fmt_size(n: int) -> str:
    if n is None:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ---- Sidebar ----
st.sidebar.header("Settings")
default_api = (
    st.secrets.get("API_URL")
    or os.environ.get("API_URL")
    or st.session_state.get("api_url", "http://localhost:8000")
)
if "api_url" not in st.session_state:
    st.session_state["api_url"] = default_api

API_URL = st.sidebar.text_input("API URL", key="api_url", value=default_api)
restrict_to_last_doc = st.sidebar.checkbox(
    "Restrict questions to last uploaded doc",
    key="restrict_last_doc",
    value=st.session_state["restrict_last_doc"],
    help="Only search chunks from the most recently uploaded document. Disables cross‑document retrieval.",
)
show_raw = st.sidebar.checkbox(
    "Show raw JSON (debug)",
    key="show_raw",
    value=st.session_state["show_raw"],
    help="Display the raw API response for debugging.",
)
show_debug = st.sidebar.checkbox(
    "Show diagnostics (collapsible)",
    key="show_debug",
    value=st.session_state["show_debug"],
    help="Show retrieval and scoring diagnostics inside a collapsed panel.",
)

if st.session_state.get("docs"):
    last = st.session_state["docs"][-1]
    st.sidebar.caption(
        f"Last doc: {last.get('name', '?')} ({last['doc_id'][:8]}…) • {_fmt_size(last.get('size', 0))}"
    )
    with st.sidebar.expander("Uploaded documents", expanded=False):
        import pandas as _pd

        rows = []
        seen_ids = set()
        for d in reversed(st.session_state["docs"]):
            did = d["doc_id"]
            if did in seen_ids:
                continue
            seen_ids.add(did)
            rows.append(
                {
                    "name": d.get("name", ""),
                    "id": did[:8] + "…",
                    "size": _fmt_size(d.get("size", 0)),
                    "doc_id": did,
                }
            )
        st.dataframe(_pd.DataFrame(rows), width="stretch", hide_index=True)

st.title("Intelligent Content Analyzer")

# ---- Upload document ----
st.header("Upload Document")
uploaded_files = st.file_uploader(
    "Choose file(s)",
    type=[
        "txt",
        "pdf",
        "docx",
        "pptx",
        "html",
        "htm",
        "md",
        "markdown",
        "csv",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
    ],
    accept_multiple_files=True,
    key="uploader",
    help="You can upload multiple files. Supported: TXT, PDF, DOCX, PPTX, HTML, MD, CSV, PNG/JPG/GIF/BMP/TIFF.",
)

if uploaded_files:
    st.write("Selected files:")
    for f in uploaded_files:
        st.caption(f"• {f.name} ({_fmt_size(getattr(f, 'size', 0))})")

if uploaded_files and st.button("Upload", key="btn_upload"):
    successes, failures, skipped = 0, [], []
    for f in uploaded_files:
        data_bytes = f.getvalue()
        file_hash = hashlib.sha1(data_bytes).hexdigest()
        if file_hash in st.session_state["doc_index"]:
            skipped.append(f.name)
            continue
        files = {"file": (f.name, data_bytes)}
        try:
            resp = requests.post(f"{API_URL}/upload_document", files=files, timeout=120)
            if resp.ok:
                data = resp.json()
                doc_id = data.get("doc_id")
                st.session_state["doc_id"] = doc_id
                entry = {
                    "doc_id": doc_id,
                    "name": f.name,
                    "size": getattr(f, "size", None),
                    "type": getattr(f, "type", None),
                    "hash": file_hash,
                }
                st.session_state["doc_index"][file_hash] = entry
                st.session_state["docs"].append(entry)
                successes += 1
            else:
                failures.append((f.name, f"{resp.status_code} {resp.text}"))
        except Exception as e:
            failures.append((f.name, str(e)))

    if successes:
        st.success(
            f"Uploaded {successes} file(s). Last doc_id: {st.session_state.get('doc_id')}"
        )
    if skipped:
        st.info("Skipped duplicate file(s): " + ", ".join(skipped))
    if failures:
        for name, err in failures:
            st.error(f"{name}: {err}")


# ---- Helper: render QA result ----
def render_qa(data: dict) -> None:
    # 1) Answer + confidence
    st.subheader("Answer")
    st.write(data.get("answer") or "No answer.")
    confidence = float(data.get("confidence") or 0.0)
    c1, c2 = st.columns([1, 4])
    with c1:
        st.metric("Confidence", f"{confidence:.2f}")
    with c2:
        st.progress(min(max(confidence, 0.0), 1.0))

    # 1b) Quality metrics (support both 'evaluation' and 'eval' keys)
    diag = data.get("diagnostics") or {}
    evalm = diag.get("evaluation") or diag.get("eval") or {}
    if evalm:
        st.subheader("Quality metrics")
        cA, cB, cC = st.columns(3)
        cA.metric("Factuality", f"{float(evalm.get('factuality', 0.0)):.2f}")
        cB.metric("Relevance", f"{float(evalm.get('relevance', 0.0)):.2f}")
        cC.metric("Completeness", f"{float(evalm.get('completeness', 0.0)):.2f}")
        cD, cE, cF = st.columns(3)
        cD.metric("Faithfulness", f"{float(evalm.get('faithfulness', 0.0)):.2f}")
        cE.metric(
            "Answer Relevance (1–5)",
            f"{float(evalm.get('answer_relevance_1_5', 0.0)):.2f}",
        )
        cF.metric(
            "Context Relevance Ratio",
            f"{float(evalm.get('context_relevance_ratio', 0.0)):.2f}",
        )

    # 2) Citations (deduped, with snippet and filename)
    st.subheader("Citations")
    citations = data.get("citations") or []
    rows, seen = [], set()
    for c in citations:
        key = c.get("chunk_id") or (c.get("doc_id"), c.get("page"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        raw_snip = c.get("snippet") or c.get("text") or c.get("section") or ""
        snip = raw_snip.strip()
        if len(snip) > 240:
            snip = snip[:240] + "…"
        rows.append(
            {
                "file": c.get("file_name") or (str(c.get("doc_id", ""))[:8] + "…"),
                "page": c.get("page"),
                "snippet": snip,
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
            }
        )
    if rows:
        df_cit = pd.DataFrame(rows)
        try:
            st.dataframe(
                df_cit[["file", "page", "snippet"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "snippet": st.column_config.TextColumn("Snippet", width="large"),
                    "file": st.column_config.TextColumn("File", width="medium"),
                    "page": st.column_config.NumberColumn("Page", width="small"),
                },
            )
        except Exception:
            st.dataframe(
                df_cit[["file", "page", "snippet"]],
                width="stretch",
                hide_index=True,
            )
    else:
        st.info("No citations returned.")

    # 3) Retrieval diagnostics (collapsed by default; only in debug mode)
    if st.session_state.get("show_debug"):
        rdiag = diag.get("retrieval") or {}
        inner = rdiag.get("diagnostics") or {}
        with st.expander("Retrieval diagnostics", expanded=False):
            cols = st.columns(4)
            cols[0].metric("Hits", rdiag.get("num_hits", 0))
            cols[1].metric(
                "Union Size", inner.get("union_size", rdiag.get("union_size", 0))
            )
            cols[2].metric("Dense Backend", rdiag.get("dense_backend", "n/a"))
            cols[3].metric("Filtered", str(rdiag.get("filtered", False)))

            top_scores = rdiag.get("top_scores") or []
            if top_scores:
                st.caption("Top scores")
                st.bar_chart(pd.DataFrame({"score": top_scores}))

            rrf_scores = inner.get("rrf_scores", {})
            bm25_scores = inner.get("bm25_scores", {})
            dense_scores = inner.get("dense_scores", {})
            bm25_rank = inner.get("bm25_rank", {})
            dense_rank = inner.get("dense_rank", {})

            if rrf_scores:
                rows = []
                for cid, rrf in sorted(
                    rrf_scores.items(), key=lambda x: x[1], reverse=True
                )[: min(25, len(rrf_scores))]:
                    rows.append(
                        {
                            "chunk_id": cid,
                            "rrf": round(float(rrf), 4),
                            "bm25": round(float(bm25_scores.get(cid, 0.0)), 4)
                            if bm25_scores
                            else None,
                            "dense": round(float(dense_scores.get(cid, 0.0)), 4)
                            if dense_scores
                            else None,
                            "bm25_rank": bm25_rank.get(cid),
                            "dense_rank": dense_rank.get(cid),
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.info("No RRF diagnostics available.")

    if st.session_state.get("show_raw"):
        with st.expander("Raw Response JSON"):
            st.json(data)


# ---- Ask question ----
st.header("Ask a Question")
with st.form("qa_form", clear_on_submit=False):
    col_q1, col_q2, col_q3, col_q4 = st.columns([3, 1, 1, 1])
    with col_q1:
        question = st.text_input(
            "Question",
            key="question",
            value=st.session_state.get("question", ""),
            placeholder="Ask something about your document...",
        )
    with col_q2:
        k = st.number_input(
            "Top-K Chunks",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("k", 10)),
            step=1,
            key="k",
            help="How many chunks to return after fusion/reranking.",
        )
    with col_q3:
        use_rerank = st.checkbox(
            "Use Reranker",
            value=bool(st.session_state.get("use_rerank", True)),
            key="use_rerank",
            help="Apply a cross‑encoder to reorder fused candidates. Improves accuracy, adds latency.",
        )
    with col_q4:
        use_judge = st.checkbox(
            "Use LLM Judge",
            value=bool(st.session_state.get("use_judge", False)),
            key="use_judge",
            help="Call an LLM to grade the answer and boost confidence. Improves reliability, adds latency.",
        )
    col_d1, _ = st.columns([1, 3])
    with col_d1:
        dense_candidates = st.number_input(
            "Dense candidates",
            min_value=10,
            max_value=500,
            step=10,
            value=int(st.session_state.get("dense_candidates", 50)),
            key="dense_candidates",
            help="Number of top results pulled from the vector index before fusion. Higher = better recall, slower.",
        )

    st.markdown("")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    payload = {
        "question": st.session_state["question"],
        "k": int(st.session_state["k"]),
        "use_rerank": bool(st.session_state["use_rerank"]),
        "use_judge": bool(st.session_state["use_judge"]),
        "dense_candidates": int(st.session_state["dense_candidates"]),
    }
    if st.session_state["restrict_last_doc"]:
        if st.session_state.get("doc_id"):
            payload["filters"] = {"include_doc_ids": [st.session_state["doc_id"]]}
        else:
            st.info("Restriction is on, but no document is uploaded yet.")
    try:
        resp = requests.post(
            f"{st.session_state['api_url']}/ask_question", json=payload, timeout=60
        )
        if resp.ok:
            st.session_state["qa_result"] = resp.json()
        else:
            st.error(f"QA failed: {resp.status_code} {resp.text}")
    except Exception as e:
        st.error(f"Request error: {e}")

if st.session_state.get("qa_result"):
    render_qa(st.session_state["qa_result"])

# ---- Document summary ----
st.header("Document Summary")
col_s1, col_s2 = st.columns([2, 1])
with col_s1:
    st.caption("Summarize the last uploaded document.")
with col_s2:
    summarize_clicked = st.button("Summarize", key="btn_summarize")

if summarize_clicked:
    doc_id = st.session_state.get("doc_id")
    if not doc_id:
        st.warning("Upload a document first.")
    else:
        try:
            resp = requests.get(
                f"{st.session_state['api_url']}/document_summary",
                params={"doc_id": doc_id},
                timeout=60,
            )
            if resp.ok:
                st.session_state["summary"] = resp.json().get("summary")
            else:
                st.error(f"Summary failed: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Summary error: {e}")

if st.session_state.get("summary"):
    st.write(st.session_state["summary"])
