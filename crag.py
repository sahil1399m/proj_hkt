"""
crag.py — HTE CRAG Pipeline — Streaming + Parallel Optimised
"""
from __future__ import annotations

import os

for _k in ["CHROMA_API_IMPL", "CHROMA_SERVER_HOST",
           "CHROMA_SERVER_HTTP_PORT", "IS_PERSISTENT"]:
    os.environ.pop(_k, None)
os.environ["ANONYMIZED_TELEMETRY"]               = "False"
os.environ["CHROMA_OTEL_EXPORTER_OTLP_ENDPOINT"] = ""

import json, logging, math, re, time, threading, urllib.request
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()
logger = logging.getLogger(__name__)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

CHROMA_PATH         = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION", "hte_documents")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
IBM_API_KEY         = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID      = os.getenv("IBM_PROJECT_ID", "")
IBM_URL             = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
IBM_MODEL_ID        = os.getenv("IBM_MODEL_ID", "ibm/granite-4-h-small")

TOP_K               = int(os.getenv("TOP_K", "6"))
CORRECT_THRESHOLD   = float(os.getenv("CORRECT_THRESHOLD", "-3.0"))
INCORRECT_THRESHOLD = float(os.getenv("INCORRECT_THRESHOLD", "-6.5"))

OFFICIAL_DOMAINS = [
    "maharashtra.gov.in", "dtemaharashtra.gov.in", "dte.maharashtra.gov.in",
    "gr.maharashtra.gov.in", "mahadbt.maharashtra.gov.in",
    "cetcell.mahacet.org", "mahacet.org", "aicte-india.org",
    "ugc.ac.in", "ugc.gov.in", "education.gov.in", "msbte.org.in",
    "scholarship.gov.in", "mahaeschol.maharashtra.gov.in",
]
TABLE_PRIORITY_INTENTS = {
    "fees", "scholarship", "admission", "hostel", "examination", "seat"
}

_cross_encoder: Optional[CrossEncoder] = None
_ce_ready = threading.Event()

def _load_ce():
    global _cross_encoder
    try:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder ready")
    except Exception as exc:
        logger.error("CrossEncoder load failed: %s", exc)
    finally:
        _ce_ready.set()

threading.Thread(target=_load_ce, daemon=True).start()

def _get_ce() -> Optional[CrossEncoder]:
    _ce_ready.wait(timeout=30)
    return _cross_encoder

# ── FIXED: lazy init, no pre-warm thread ──────────────────────────────────────
_chroma_col = None
_col_lock   = threading.Lock()

def _get_col():
    global _chroma_col
    if _chroma_col is not None:
        return _chroma_col
    with _col_lock:
        if _chroma_col is None:
            try:
                client      = chromadb.PersistentClient(path=CHROMA_PATH)
                _chroma_col = client.get_collection(CHROMA_COLLECTION)
                logger.info("ChromaDB ready — %d chunks", _chroma_col.count())
            except Exception as exc:
                logger.error("ChromaDB init failed: %s", exc)
    return _chroma_col

def _reset_col():
    """Force ChromaDB re-init on next query (called by app.py after HF download)."""
    global _chroma_col
    with _col_lock:
        _chroma_col = None
    logger.info("ChromaDB singleton reset")

# NO threading.Thread for _get_col — lazy init only

_granite_model = None
_granite_lock  = threading.Lock()

def _get_granite():
    global _granite_model
    if _granite_model is not None:
        return _granite_model
    with _granite_lock:
        if _granite_model is None:
            if not (IBM_API_KEY and IBM_PROJECT_ID):
                return None
            try:
                from ibm_watsonx_ai import Credentials
                from ibm_watsonx_ai.foundation_models import ModelInference
                _granite_model = ModelInference(
                    model_id=IBM_MODEL_ID,
                    credentials=Credentials(api_key=IBM_API_KEY, url=IBM_URL),
                    project_id=IBM_PROJECT_ID,
                    params={"max_tokens": 1200, "temperature": 0.05},
                )
                logger.info("IBM Granite cached")
            except Exception as exc:
                logger.error("IBM Granite init failed: %s", exc)
    return _granite_model

@dataclass
class RetrievedChunk:
    content: str
    source: str
    page: int
    category: str
    score: float      = 0.0
    chunk_id: str     = ""
    element_type: str = "text"
    language: str     = "en"
    source_type: str  = "pdf"

@dataclass
class WebResult:
    title: str
    url: str
    content: str
    score: float = 0.0

@dataclass
class CRAGResult:
    answer: str
    confidence_label: str
    confidence_score: float
    doc_sources: list[RetrievedChunk]
    web_sources: list[WebResult]
    pipeline_trace: dict[str, Any]
    translation_applied: bool = False
    translated_answer: str    = ""
    model_used: str           = "granite-4-h-small"
    language_out: str         = "english"
    error: Optional[str]      = None
    total_time_s: float       = 0.0
    selected_agents: list     = field(default_factory=list)

_INTENT_MAP = [
    ("fees",        ["fee", "fees", "tuition", "charges", "amount", "cost", "fra", "payment", "refund"]),
    ("scholarship", ["scholarship", "freeship", "ebc", "obc", "sc", "st", "vjnt", "mahadbt", "stipend", "fellowship"]),
    ("admission",   ["admission", "cap", "merit", "cutoff", "seat", "allotment", "mht-cet", "cet", "apply", "eligibility", "registration"]),
    ("hostel",      ["hostel", "accommodation", "dormitory", "room", "boarding", "mess"]),
    ("examination", ["exam", "examination", "result", "marksheet", "backlog", "atkt", "revaluation", "hall ticket"]),
    ("affiliation", ["affiliation", "recognition", "approval", "noc", "aicte", "ugc"]),
    ("circular",    ["circular", "notice", "notification", "order", "gr ", "government resolution", "gazette"]),
    ("regulation",  ["regulation", "rules", "act ", "statute", "ordinance", "policy"]),
    ("placement",   ["placement", "campus", "recruit", "job"]),
    ("curriculum",  ["curriculum", "syllabus", "course", "subject", "semester"]),
]

def _detect_intent(query: str) -> str:
    q = query.lower()
    for intent, keywords in _INTENT_MAP:
        if any(kw in q for kw in keywords):
            return intent
    return "general"

def _detect_language(query: str) -> str:
    return "mr" if sum(1 for c in query if "\u0900" <= c <= "\u097f") > 2 else "en"

def _needs_live_data(query: str) -> bool:
    q = query.lower()
    return any(w in q for w in [
        "today", "current", "now", "deadline", "last date",
        "closing", "open", "status", "2025", "2026",
    ])

def embed_query(text: str) -> list[float]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-001:embedContent?key={GOOGLE_API_KEY}"
    )
    payload = json.dumps({
        "model":    "models/gemini-embedding-001",
        "content":  {"parts": [{"text": text[:6000]}]},
        "taskType": "RETRIEVAL_QUERY",
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["embedding"]["values"]

def _make_chunk(doc: str, meta: dict, dist: float, et: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=doc,
        source=meta.get("source") or meta.get("filename") or "Unknown",
        page=int(meta.get("page", 0)) if meta.get("page") is not None else 0,
        category=meta.get("category", ""),
        score=float(1 - dist),
        chunk_id=meta.get("chunk_id") or meta.get("filename") or doc[:40],
        element_type=et if et else meta.get("element_type", "text"),
        language=meta.get("language", "en"),
        source_type=meta.get("source_type", "pdf"),
    )

def retrieve_from_chroma(embedding: list[float], intent: str) -> list[RetrievedChunk]:
    col = _get_col()
    if col is None:
        return []
    try:
        total  = col.count()
        if total == 0:
            return []
        safe_k = min(TOP_K, total)
        chunks: list[RetrievedChunk] = []
        seen:   set[str] = set()

        sample = col.get(limit=1, include=["metadatas"])
        meta0  = sample["metadatas"][0] if sample["metadatas"] else {}
        has_element_type = "element_type" in meta0
        has_language     = "language" in meta0

        if has_element_type and intent in TABLE_PRIORITY_INTENTS:
            try:
                tr = col.query(
                    query_embeddings=[embedding],
                    n_results=min(3, total),
                    where={"element_type": {"$eq": "table"}},
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(
                    tr["documents"][0], tr["metadatas"][0], tr["distances"][0]
                ):
                    cid = meta.get("chunk_id", meta.get("filename", doc[:40]))
                    if cid not in seen:
                        seen.add(cid)
                        chunks.append(_make_chunk(doc, meta, dist, "table"))
            except Exception as exc:
                logger.warning("Table-priority query failed: %s", exc)

        if has_language and intent != "general":
            try:
                mr = col.query(
                    query_embeddings=[embedding],
                    n_results=min(2, total),
                    where={"language": {"$eq": "mr"}},
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist in zip(
                    mr["documents"][0], mr["metadatas"][0], mr["distances"][0]
                ):
                    cid = meta.get("chunk_id", meta.get("filename", doc[:40]))
                    if cid not in seen:
                        seen.add(cid)
                        chunks.append(_make_chunk(doc, meta, dist,
                                                  meta.get("element_type", "text")))
            except Exception as exc:
                logger.warning("Marathi query failed: %s", exc)

        rem = max(safe_k - len(chunks), safe_k)
        res = col.query(
            query_embeddings=[embedding],
            n_results=min(rem, total),
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            cid = meta.get("chunk_id", meta.get("filename", doc[:40]))
            if cid not in seen:
                seen.add(cid)
                chunks.append(_make_chunk(doc, meta, dist,
                                          meta.get("element_type", "text")))

        logger.info("Retrieved %d chunks (%d tables)", len(chunks),
                    sum(1 for c in chunks if c.element_type == "table"))
        return chunks
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        return []

_TABLE_KWS = {
    "fee", "fees", "tuition", "scholarship", "seat", "amount",
    "hostel", "marks", "percentage", "sc", "st", "obc", "cutoff"
}

def rerank_chunks(query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], float]:
    if not chunks:
        return [], -10.0
    chunks = chunks[:8]
    ce = _get_ce()
    if ce is None:
        return chunks, -5.0
    pairs = [(query, c.content) for c in chunks]
    try:
        scores = ce.predict(pairs).tolist()
    except Exception as exc:
        logger.error("CrossEncoder error: %s", exc)
        return chunks, -5.0
    is_tq = any(kw in query.lower() for kw in _TABLE_KWS)
    for chunk, score in zip(chunks, scores):
        chunk.score = score + (0.5 if chunk.element_type == "table" and is_tq else 0.0)
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks, chunks[0].score

def classify_confidence(logit: float) -> tuple[str, float]:
    norm = 1 / (1 + math.exp(-logit / 2))
    if logit >= CORRECT_THRESHOLD:
        return "HIGH", round(norm, 4)
    elif logit <= INCORRECT_THRESHOLD:
        return "LOW", round(norm, 4)
    return "MEDIUM", round(norm, 4)

def search_web(query: str, max_results: int = 4) -> list[WebResult]:
    if not TAVILY_API_KEY:
        return []
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        resp    = client.search(
            query=query, max_results=max_results,
            search_depth="basic", include_domains=OFFICIAL_DOMAINS,
        )
        results = [
            WebResult(title=r.get("title", ""), url=r.get("url", ""),
                      content=r.get("content", ""), score=r.get("score", 0.0))
            for r in resp.get("results", [])
        ]
        if len(results) < 2:
            resp2 = client.search(query=query, max_results=max_results, search_depth="basic")
            seen  = {r.url for r in results}
            for r in resp2.get("results", []):
                if r.get("url", "") not in seen:
                    results.append(WebResult(title=r.get("title", ""), url=r.get("url", ""),
                                             content=r.get("content", ""), score=r.get("score", 0.0)))
        logger.info("Tavily: %d results", len(results))
        return results[:max_results]
    except Exception as exc:
        logger.warning("Tavily error: %s", exc)
        return []

def build_context(doc_chunks: list[RetrievedChunk], web_results: list[WebResult],
                  max_chars: int = 7000) -> str:
    parts: list[str] = []
    chars = 0
    for i, c in enumerate([x for x in doc_chunks if x.element_type == "table"][:3]):
        blk = f"[TABLE {i+1}] {c.source} p.{c.page} | {c.category}\n```\n{c.content}\n```\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)
    for i, c in enumerate([x for x in doc_chunks if x.element_type != "table"][:4]):
        blk = f"[DOC {i+1}] {c.source} p.{c.page} | {c.category}\n{c.content}\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)
    for i, w in enumerate(web_results[:3]):
        blk = f"[WEB {i+1}] {w.title}\nURL: {w.url}\n{w.content}\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)
    return "\n---\n".join(parts)

SYSTEM_PROMPT = """You are an official AI assistant for Maharashtra's Higher & Technical Education (HTE) Department.

RULES — follow every rule:
1. Answer ONLY from the provided context. Never invent facts.
2. If context lacks the answer: "I could not find this in official HTE documents. Please contact the department at https://www.dtemaharashtra.gov.in"
3. Cite sources as [TABLE N], [DOC N], or [WEB N] after every factual claim.
4. Reproduce markdown tables from context as formatted tables — never flatten them.
5. Quote exact numbers for fees, scholarships, seat counts, dates.
6. Use ## headings for sections, bullet points for lists of 3+.
7. Start your answer directly — no preamble, no "Sure!", no "Great question!".
8. End cleanly — no "I hope this helps"."""

def _build_user_msg(query: str, context: str, intent: str, conf: str) -> str:
    return (
        f"Question: {query}\nIntent: {intent}\nDoc confidence: {conf}\n\n"
        f"Context from official HTE sources:\n{context}\n\nAnswer:"
    )

def _clean(text: str) -> str:
    if not text:
        return text
    for pat in [
        r"^.*?(?:assistant\s*final|<\|assistant\|>)\s*",
        r"^assistantfinal\s*",
        r"^assistant\s*",
    ]:
        c = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
        if c != text:
            text = c.strip(); break
    text = re.sub(r"\n{0,3}(#{1,3}\s*)?(Sources?|References?|Citations?)\s*\n.*$",
                  "", text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def stream_groq(query: str, context: str, intent: str, conf: str) -> Generator[str, None, None]:
    if not GROQ_API_KEY:
        yield "Error: GROQ_API_KEY not configured."
        return
    client = Groq(api_key=GROQ_API_KEY)
    try:
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_msg(query, context, intent, conf)},
            ],
            max_tokens=1000, temperature=0.05, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception as exc:
        logger.error("Groq stream error: %s", exc)
        yield f"\n\n⚠️ Generation error: {exc}"

def _generate_groq(query: str, context: str, intent: str, conf: str) -> tuple[str, str]:
    if not GROQ_API_KEY:
        return "", ""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_msg(query, context, intent, conf)},
            ],
            max_tokens=1000, temperature=0.05, stream=False,
        )
        answer = _clean(resp.choices[0].message.content)
        return (answer, "groq-llama-3.3-70b") if answer and len(answer) > 20 else ("", "")
    except Exception as exc:
        logger.warning("Groq error: %s", exc)
        return "", ""

def _generate_ibm(query: str, context: str, intent: str, conf: str) -> tuple[str, str]:
    model = _get_granite()
    if model is None:
        return "", ""
    try:
        resp    = model.chat(messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_msg(query, context, intent, conf)},
        ])
        choices = resp.get("choices", [])
        if choices:
            msg    = choices[0].get("message", {})
            answer = _clean(msg.get("content") or msg.get("text") or "")
            return (answer, IBM_MODEL_ID.split("/")[-1]) if answer and len(answer) > 20 else ("", "")
    except Exception as exc:
        if "429" in str(exc) or "Too Many Requests" in str(exc):
            logger.warning("IBM rate limited")
        else:
            logger.warning("IBM error: %s", exc)
    return "", ""

def translate_answer(answer: str, target_language: str) -> str:
    if target_language == "english" or not answer.strip():
        return ""
    lang_name = (
        "Marathi (Devanagari script)" if target_language == "marathi"
        else "Hindi (Devanagari script)"
    )
    prompt = (
        f"Translate the following official Maharashtra government education answer "
        f"from English to {lang_name}.\n"
        f"RULES: preserve all markdown, citation markers [DOC N]/[TABLE N]/[WEB N], "
        f"numbers, dates, proper nouns.\n\n"
        f"English:\n---\n{answer}\n---\n\n{lang_name} translation:"
    )
    def _gemini(p):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.0-flash:generateContent?key={GOOGLE_API_KEY}"
        )
        body = json.dumps({"contents": [{"parts": [{"text": p}]}],
                           "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}}).encode()
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["candidates"][0]["content"]["parts"][0]["text"].strip()
    def _groq_tr(p):
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": p}],
            max_tokens=2048, temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    for fn in [_gemini, _groq_tr]:
        try:
            t = fn(prompt)
            if t and len(t) > 20:
                return t
        except Exception as exc:
            logger.warning("Translation via %s failed: %s", fn.__name__, exc)
    return ""

def build_citations(doc_chunks: list[RetrievedChunk], web_results: list[WebResult]) -> dict:
    return {
        "documents": [
            {"index": i+1, "source": c.source, "page": c.page,
             "category": c.category, "score": round(c.score, 3),
             "element_type": c.element_type, "language": c.language,
             "source_type": c.source_type}
            for i, c in enumerate(doc_chunks[:6])
        ],
        "web": [
            {"index": i+1, "title": w.title, "url": w.url, "score": round(w.score, 3)}
            for i, w in enumerate(web_results[:5])
        ],
    }

def _run_retrieval(query: str) -> dict[str, Any]:
    t0     = time.time()
    intent = _detect_intent(query)
    lang   = _detect_language(query)
    live   = _needs_live_data(query)
    try:
        embedding = embed_query(query)
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        return {"ok": False, "error": str(exc), "intent": intent}
    doc_chunks         = retrieve_from_chroma(embedding, intent)
    ranked, best_logit = rerank_chunks(query, doc_chunks)
    conf_label, conf_score = classify_confidence(best_logit)
    web_results: list[WebResult] = []
    used_docs = ranked
    if conf_label == "HIGH" and not live:
        branch = "local_docs_only"
    elif conf_label == "MEDIUM" or live:
        web_results = search_web(query)
        branch      = f"docs_plus_web ({len(web_results)} results)"
    else:
        used_docs   = ranked[:4] if best_logit > INCORRECT_THRESHOLD else []
        web_results = search_web(query)
        branch      = f"web_only ({len(web_results)} results)"
        if used_docs:
            conf_label = "MEDIUM"
            branch     = f"weak_docs_plus_web ({len(web_results)} results)"
    context = build_context(used_docs, web_results)
    return {
        "ok": True, "intent": intent, "language": lang,
        "used_docs": used_docs, "web_results": web_results,
        "context": context, "conf_label": conf_label, "conf_score": conf_score,
        "branch": branch, "retrieval_s": round(time.time() - t0, 2),
        "trace": {
            "query":      {"intent": intent, "language": lang, "needs_live": live},
            "retrieval":  {"chunks": len(doc_chunks),
                           "tables": sum(1 for c in doc_chunks if c.element_type == "table")},
            "crossencoder": {"best_logit": round(best_logit, 3),
                             "confidence_label": conf_label, "confidence_score": conf_score},
            "corrective_branch": {"action": branch},
            "context": {"chars": len(context)},
        },
    }


def stream_crag_pipeline(
    query: str,
    language_out: str = "english",
) -> Generator[str, None, None]:
    """
    Streaming pipeline.
    PRIMARY  — IBM Granite (generates full answer, yields word-by-word)
    FALLBACK — Groq Llama  (true token streaming)
    """
    import time as _st

    yield '<<<STATUS>>>' + json.dumps({"stage": "embedding", "msg": "Generating semantic embeddings…"})
    r = _run_retrieval(query)

    if not r["ok"]:
        yield '<<<STATUS>>>' + json.dumps({"stage": "error", "msg": "Retrieval failed"})
        yield f"❌ Retrieval error: {r.get('error', 'unknown')}"
        yield '\n<<<META>>>' + json.dumps({"error": r.get("error")})
        return

    has_web       = len(r["web_results"]) > 0
    retrieved_msg = f"Found {len(r['used_docs'])} doc chunks"
    if has_web:
        retrieved_msg += f" + {len(r['web_results'])} web results"
    yield '<<<STATUS>>>' + json.dumps({"stage": "retrieved", "msg": retrieved_msg})

    context    = r["context"]
    conf_label = r["conf_label"]
    intent     = r["intent"]
    model_used = "none"
    full_answer: list[str] = []

    if not context.strip():
        yield (
            "I could not find relevant information in official HTE documents "
            "or government websites. Please contact Maharashtra HTE at "
            "https://www.dtemaharashtra.gov.in"
        )
    else:
        # ── PRIMARY: IBM Granite ──────────────────────────────────────────
        yield '<<<STATUS>>>' + json.dumps({
            "stage": "generating",
            "msg":   "IBM Granite is generating your answer…"
        })
        ibm_answer, _ = _generate_ibm(query, context, intent, conf_label)

        if ibm_answer:
            model_used = IBM_MODEL_ID.split("/")[-1]
            # Word-chunk streaming so the UI cursor animates
            words = ibm_answer.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                full_answer.append(token)
                yield token
                _st.sleep(0.008)
        else:
            # ── FALLBACK: Groq true streaming ─────────────────────────────
            logger.warning("IBM Granite empty in stream path — falling back to Groq")
            yield '<<<STATUS>>>' + json.dumps({
                "stage": "generating",
                "msg":   "Groq Llama generating answer (IBM fallback)…"
            })
            model_used = "groq-llama-3.3-70b"
            for token in stream_groq(query, context, intent, conf_label):
                full_answer.append(token)
                yield token

        if not "".join(full_answer).strip():
            yield "Unable to generate answer. Please try again."
            model_used = "none"

    yield '<<<STATUS>>>' + json.dumps({"stage": "done", "msg": "Done"})
    meta = {
        "confidence_label": r["conf_label"],
        "confidence_score": r["conf_score"],
        "model_used":       model_used,
        "citations":        build_citations(r["used_docs"], r["web_results"]),
        "pipeline_trace":   r["trace"],
        "retrieval_s":      r["retrieval_s"],
        "language_out":     language_out,
    }
    yield '\n<<<META>>>' + json.dumps(meta)


def run_crag_pipeline(
    query: str,
    language_out: str = "english",
    translate_to_marathi: bool = False,
) -> CRAGResult:
    t0 = time.time()
    if translate_to_marathi and language_out == "english":
        language_out = "marathi"
    r = _run_retrieval(query)
    if not r["ok"]:
        return CRAGResult(
            answer=f"Retrieval error: {r.get('error')}",
            confidence_label="LOW", confidence_score=0.0,
            doc_sources=[], web_sources=[],
            pipeline_trace={"error": r.get("error")},
            error=r.get("error"),
        )
    context    = r["context"]
    conf_label = r["conf_label"]
    intent     = r["intent"]
    if not context.strip():
        answer     = ("I could not find relevant information. Please contact "
                      "Maharashtra HTE at https://www.dtemaharashtra.gov.in")
        model_used = "none"
    else:
        # ── IBM Granite first ─────────────────────────────────────────────
        answer, model_used = _generate_ibm(query, context, intent, conf_label)
        if not answer:
            logger.warning("IBM Granite returned empty — falling back to Groq")
            answer, model_used = _generate_groq(query, context, intent, conf_label)
        if not answer:
            answer     = "Unable to generate answer. Please check API keys and try again."
            model_used = "none"
    translated_answer   = ""
    translation_applied = False
    if language_out in ("marathi", "hindi") and answer and model_used != "none":
        translated_answer   = translate_answer(answer, language_out)
        translation_applied = bool(translated_answer)
        r["trace"]["translation"] = {"target": language_out, "applied": translation_applied}
    total_t = round(time.time() - t0, 2)
    r["trace"]["llm_generation"] = {"model": model_used, "chars": len(answer)}
    r["trace"]["total_time_s"]   = total_t
    return CRAGResult(
        answer=answer, confidence_label=conf_label,
        confidence_score=r["conf_score"],
        doc_sources=r["used_docs"][:6], web_sources=r["web_results"][:5],
        pipeline_trace=r["trace"], translation_applied=translation_applied,
        translated_answer=translated_answer, model_used=model_used,
        language_out=language_out, total_time_s=total_t,
    )
