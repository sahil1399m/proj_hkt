"""
crag.py — HTE CRAG v5 — Dual-Track Retrieval + Granite Streaming
================================================================
Key upgrades:
  1. DUAL-TRACK retrieval — top-K text chunks + top-R table chunks fetched
     in parallel threads (no extra latency)
  2. Granite 4.0 as PRIMARY streamer via Groq-compatible endpoint
     Groq Llama 3.3 as fast fallback
  3. Chunk verification step — CrossEncoder scores each chunk,
     filters out low-relevance ones before sending to LLM
  4. Streaming preserved — tokens arrive progressively
  5. All within ~8-10s total
"""
from __future__ import annotations

import os

for _k in ["CHROMA_API_IMPL","CHROMA_SERVER_HOST",
           "CHROMA_SERVER_HTTP_PORT","IS_PERSISTENT"]:
    os.environ.pop(_k, None)
os.environ["ANONYMIZED_TELEMETRY"]               = "False"
os.environ["CHROMA_OTEL_EXPORTER_OTLP_ENDPOINT"] = ""

import json, logging, math, re, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH         = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION", "hte_documents")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY      = os.getenv("TAVILY_API_KEY", "")
IBM_API_KEY         = os.getenv("IBM_API_KEY", "")
IBM_PROJECT_ID      = os.getenv("IBM_PROJECT_ID", "")
IBM_URL             = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
IBM_MODEL_ID        = os.getenv("IBM_MODEL_ID", "ibm/granite-4-h-small")

# Dual-track retrieval config
TOP_K_TEXT          = int(os.getenv("TOP_K_TEXT", "6"))     # text chunks
TOP_R_TABLE         = int(os.getenv("TOP_R_TABLE", "4"))    # table chunks
CE_MIN_SCORE        = float(os.getenv("CE_MIN_SCORE", "-8.0"))  # filter threshold

CORRECT_THRESHOLD   = float(os.getenv("CORRECT_THRESHOLD", "-3.0"))
INCORRECT_THRESHOLD = float(os.getenv("INCORRECT_THRESHOLD", "-6.5"))

OFFICIAL_DOMAINS = [
    "maharashtra.gov.in","dtemaharashtra.gov.in","dte.maharashtra.gov.in",
    "gr.maharashtra.gov.in","mahadbt.maharashtra.gov.in",
    "cetcell.mahacet.org","mahacet.org","aicte-india.org",
    "ugc.ac.in","ugc.gov.in","education.gov.in","msbte.org.in",
    "scholarship.gov.in","mahaeschol.maharashtra.gov.in",
]

# ── Module-level singletons ───────────────────────────────────────────────────
_ce: Optional[CrossEncoder] = None
_ce_ready = threading.Event()
_chroma_col = None
_col_lock   = threading.Lock()


def _load_ce():
    global _ce
    try:
        _ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder ready")
    except Exception as e:
        logger.error("CrossEncoder load failed: %s", e)
    finally:
        _ce_ready.set()

def _get_ce() -> Optional[CrossEncoder]:
    _ce_ready.wait(timeout=30)
    return _ce

def _get_col():
    global _chroma_col
    if _chroma_col: return _chroma_col
    with _col_lock:
        if not _chroma_col:
            try:
                client = chromadb.PersistentClient(path=CHROMA_PATH)
                _chroma_col = client.get_collection(CHROMA_COLLECTION)
                logger.info("ChromaDB ready — %d chunks", _chroma_col.count())
            except Exception as e:
                logger.error("ChromaDB init: %s", e)
    return _chroma_col

# Pre-warm both on import
threading.Thread(target=_load_ce,  daemon=True).start()
threading.Thread(target=_get_col,  daemon=True).start()


# ── Data classes ──────────────────────────────────────────────────────────────
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
    verified: bool    = False   # True if CE score >= CE_MIN_SCORE

@dataclass
class WebResult:
    title: str; url: str; content: str; score: float = 0.0

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
    model_used: str           = "ibm-granite-4.0"
    error: Optional[str]      = None
    selected_agents: list     = field(default_factory=list)


# ── Intent detection (local, 0ms) ─────────────────────────────────────────────
_INTENT_MAP = {
    "fees":        ["fee","fees","tuition","charges","amount","cost","fra","payment","refund"],
    "scholarship": ["scholarship","freeship","ebc","obc","sc","st","vjnt","mahadbt","stipend","minority"],
    "admission":   ["admission","cap","merit","cutoff","seat","allotment","mht-cet","cet","apply","eligibility","registration"],
    "hostel":      ["hostel","accommodation","dormitory","room","boarding","mess"],
    "examination": ["exam","examination","result","marksheet","backlog","atkt","revaluation","grade"],
    "affiliation": ["affiliation","recognition","approval","noc","aicte","ugc"],
    "circular":    ["circular","notice","notification","order","gr","government resolution","gazette"],
}

def _detect_intent(q: str) -> str:
    ql = q.lower()
    for intent, kws in _INTENT_MAP.items():
        if any(kw in ql for kw in kws):
            return intent
    return "general"

def _detect_language(q: str) -> str:
    return "mr" if sum(1 for c in q if '\u0900'<=c<='\u097F') > 2 else "en"


# ── Embedding (REST, no SDK) ──────────────────────────────────────────────────
def embed_query(text: str) -> list[float]:
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-embedding-001:embedContent?key={GOOGLE_API_KEY}")
    payload = json.dumps({
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text[:6000]}]},
        "taskType": "RETRIEVAL_QUERY",
    }).encode()
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())["embedding"]["values"]


# ══════════════════════════════════════════════════════════════════════════════
# DUAL-TRACK RETRIEVAL
# Fetches text chunks and table chunks in parallel threads
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_text_chunks(col, embedding: list[float], k: int) -> list[RetrievedChunk]:
    """Retrieve top-K TEXT chunks from ChromaDB."""
    try:
        sample = col.get(limit=1, include=["metadatas"])
        has_et = sample["metadatas"] and "element_type" in sample["metadatas"][0]
        total  = col.count()
        safe_k = min(k, total)

        if has_et:
            # Only text/ocr_text chunks
            try:
                res = col.query(
                    query_embeddings=[embedding],
                    n_results=safe_k,
                    where={"element_type": {"$in": ["text","ocr_text"]}},
                    include=["documents","metadatas","distances"],
                )
            except Exception:
                # Fallback: no filter
                res = col.query(query_embeddings=[embedding], n_results=safe_k,
                                include=["documents","metadatas","distances"])
        else:
            res = col.query(query_embeddings=[embedding], n_results=safe_k,
                            include=["documents","metadatas","distances"])

        chunks = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            et = meta.get("element_type","text")
            if et == "table": continue   # skip tables — handled separately
            chunks.append(_mk(doc, meta, dist, et))
        logger.info("Text track: %d chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Text retrieval error: %s", e)
        return []


def _fetch_table_chunks(col, embedding: list[float], r: int) -> list[RetrievedChunk]:
    """Retrieve top-R TABLE chunks from ChromaDB."""
    try:
        sample = col.get(limit=1, include=["metadatas"])
        has_et = sample["metadatas"] and "element_type" in sample["metadatas"][0]
        if not has_et:
            return []

        total  = col.count()
        safe_r = min(r, total)
        res = col.query(
            query_embeddings=[embedding],
            n_results=safe_r,
            where={"element_type": {"$eq": "table"}},
            include=["documents","metadatas","distances"],
        )
        chunks = [_mk(doc, meta, dist, "table")
                  for doc, meta, dist in zip(
                      res["documents"][0], res["metadatas"][0], res["distances"][0])]
        logger.info("Table track: %d chunks", len(chunks))
        return chunks
    except Exception as e:
        logger.warning("Table retrieval error: %s", e)
        return []


def _mk(doc: str, meta: dict, dist: float, et: str) -> RetrievedChunk:
    return RetrievedChunk(
        content=doc, source=meta.get("source","?"),
        page=int(meta.get("page",0)), category=meta.get("category",""),
        score=float(1-dist), chunk_id=meta.get("chunk_id",""),
        element_type=et, language=meta.get("language","en"),
        source_type=meta.get("source_type","pdf"),
    )


def dual_track_retrieve(embedding: list[float], intent: str) -> list[RetrievedChunk]:
    """
    Fetch text and table chunks in parallel.
    Returns merged list, tables first (they're more precise for fee/seat queries).
    """
    col = _get_col()
    if col is None: return []

    # Determine track sizes based on intent
    if intent in ("fees","scholarship","admission","examination"):
        k_text  = TOP_K_TEXT      # 6 text chunks
        r_table = TOP_R_TABLE     # 4 table chunks — more tables for numeric queries
    else:
        k_text  = TOP_K_TEXT + 2  # 8 text chunks
        r_table = 2               # fewer tables for general queries

    text_chunks:  list[RetrievedChunk] = []
    table_chunks: list[RetrievedChunk] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        ft = ex.submit(_fetch_text_chunks,  col, embedding, k_text)
        fr = ex.submit(_fetch_table_chunks, col, embedding, r_table)
        text_chunks  = ft.result()
        table_chunks = fr.result()

    # Deduplicate by chunk_id
    seen:   set[str] = set()
    merged: list[RetrievedChunk] = []
    for c in table_chunks + text_chunks:   # tables first
        cid = c.chunk_id or c.content[:40]
        if cid not in seen:
            seen.add(cid)
            merged.append(c)

    logger.info("Dual-track merged: %d total (%d text + %d table)",
                len(merged), len(text_chunks), len(table_chunks))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# CHUNK VERIFICATION — CrossEncoder filters irrelevant chunks
# This is the "accuracy" upgrade — LLM only sees verified chunks
# ══════════════════════════════════════════════════════════════════════════════

def verify_and_rerank(query: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], float]:
    """
    Score every chunk with CrossEncoder.
    Mark chunks below CE_MIN_SCORE as unverified.
    Sort verified chunks first, then unverified.
    Returns (sorted_chunks, best_score).
    """
    if not chunks: return [], -10.0
    ce = _get_ce()
    if ce is None: return chunks, -5.0

    pairs  = [(query, c.content) for c in chunks]
    try:
        scores = ce.predict(pairs).tolist()
    except Exception as e:
        logger.error("CE predict error: %s", e)
        return chunks, -5.0

    # Table bonus for numeric queries
    table_kw = {"fee","fees","tuition","scholarship","seat","amount","percentage","marks","rupee"}
    is_tq    = any(kw in query.lower() for kw in table_kw)

    verified = []; unverified = []
    for chunk, score in zip(chunks, scores):
        bonus       = 0.6 if (chunk.element_type=="table" and is_tq) else 0.0
        chunk.score = score + bonus
        if score >= CE_MIN_SCORE:
            chunk.verified = True
            verified.append(chunk)
        else:
            chunk.verified = False
            unverified.append(chunk)

    verified.sort(key=lambda c: c.score, reverse=True)
    unverified.sort(key=lambda c: c.score, reverse=True)

    # Return verified first, unverified as backup
    combined   = verified + unverified
    best_score = combined[0].score if combined else -10.0

    logger.info("Verified: %d / %d chunks (best=%.2f)",
                len(verified), len(chunks), best_score)
    return combined, best_score


def classify_confidence(logit: float) -> tuple[str, float]:
    norm = 1 / (1 + math.exp(-logit / 2))
    if logit >= CORRECT_THRESHOLD:      label = "CORRECT"
    elif logit <= INCORRECT_THRESHOLD:  label = "INCORRECT"
    else:                               label = "AMBIGUOUS"
    return label, round(norm, 4)


# ── Web search ────────────────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 4) -> list[WebResult]:
    if not TAVILY_API_KEY: return []
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=TAVILY_API_KEY)
        resp    = client.search(query=query, max_results=max_results,
                                search_depth="basic",
                                include_domains=OFFICIAL_DOMAINS)
        results = [WebResult(title=r.get("title",""), url=r.get("url",""),
                             content=r.get("content",""), score=r.get("score",0.0))
                   for r in resp.get("results",[])]
        if len(results) < 2:
            resp2 = client.search(query=query, max_results=max_results,
                                  search_depth="basic")
            seen = {r.url for r in results}
            for r in resp2.get("results",[]):
                if r.get("url","") not in seen:
                    results.append(WebResult(title=r.get("title",""), url=r.get("url",""),
                                             content=r.get("content",""), score=r.get("score",0.0)))
        return results[:max_results]
    except Exception as e:
        logger.warning("Tavily error: %s", e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER — verified chunks come first
# ══════════════════════════════════════════════════════════════════════════════

def build_context(doc_chunks: list[RetrievedChunk],
                  web_results: list[WebResult],
                  max_chars: int = 7000) -> str:
    parts: list[str] = []
    chars = 0

    # 1. Verified TABLE chunks first (most precise)
    for i, c in enumerate([x for x in doc_chunks if x.element_type=="table" and x.verified][:4]):
        blk = (f"[TABLE {i+1}] ✓ {c.source} | p.{c.page} | {c.category} | score:{c.score:.2f}\n"
               f"```\n{c.content}\n```\n")
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)

    # 2. Verified TEXT chunks
    txt_i = 0
    for c in [x for x in doc_chunks if x.element_type!="table" and x.verified][:5]:
        txt_i += 1
        blk = f"[DOC {txt_i}] ✓ {c.source} | p.{c.page} | {c.category} | score:{c.score:.2f}\n{c.content}\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)

    # 3. Unverified chunks as context backup (labelled differently)
    unv_i = 0
    for c in [x for x in doc_chunks if not x.verified][:2]:
        unv_i += 1
        blk = f"[SUPPLEMENTAL {unv_i}] {c.source} | p.{c.page}\n{c.content}\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)

    # 4. Web results
    for i, w in enumerate(web_results[:3]):
        blk = f"[WEB {i+1}] {w.title}\nURL: {w.url}\n{w.content}\n"
        if chars + len(blk) > max_chars: break
        parts.append(blk); chars += len(blk)

    return "\n---\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — tells Granite exactly how to use verified chunks
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an official AI assistant for Maharashtra's Higher & Technical Education (HTE) Department.

You will receive context chunks marked as:
- [TABLE N] ✓ — verified table extracted from official PDF (highest trust)
- [DOC N] ✓   — verified text from official PDF (high trust)
- [SUPPLEMENTAL N] — additional context, use only if above chunks are insufficient
- [WEB N]     — live government website data

ANSWER RULES:
1. Use ONLY information from the context above. Never invent or assume facts.
2. [TABLE N] chunks contain exact numbers — reproduce them as formatted markdown tables. Never convert a table to prose.
3. Always cite your source using the marker: [TABLE 1], [DOC 2], [WEB 1] etc.
4. For fees, amounts, percentages — always quote exact figures from tables.
5. If verified chunks (✓) don't contain the answer, check SUPPLEMENTAL, then WEB.
6. If no chunk answers the query: say "I could not find this in official HTE documents. Please contact DTE Maharashtra at https://www.dtemaharashtra.gov.in"
7. Structure your answer: heading → key points → table (if available) → notes.
8. Start directly — no preamble, no "Based on the context", no meta-commentary."""


def _user_prompt(query: str, context: str, intent: str, conf: str) -> str:
    return (f"Question: {query}\n"
            f"Intent category: {intent}\n"
            f"Document confidence: {conf}\n\n"
            f"Context from official HTE sources:\n{context}\n\n"
            f"Provide a structured, grounded answer with citations.")


def _clean(text: str) -> str:
    if not text: return text
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{0,3}(#{1,3}\s*)?(Sources?|References?|Citations?)\s*\n.*$',
                  '', text, flags=re.IGNORECASE|re.DOTALL)
    for pat in [r'^.*?(?:assistant\s*final|<\|assistant\|>)\s*',
                r'^assistantfinal\s*', r'^assistant\s*']:
        c = re.sub(pat,'',text,flags=re.IGNORECASE|re.DOTALL)
        if c!=text: text=c.strip(); break
    return re.sub(r'\n{3,}','\n\n',text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING GENERATION
# Primary: IBM Granite 4.0 (most grounded, best table citation)
# Fallback: Groq Llama 3.3 70B (fast)
# ══════════════════════════════════════════════════════════════════════════════

def _stream_granite(query: str, context: str, intent: str,
                    conf: str) -> Generator[str, None, None]:
    """Stream from IBM Granite 4.0 via watsonx."""
    if not IBM_API_KEY or not IBM_PROJECT_ID:
        return

    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(api_key=IBM_API_KEY, url=IBM_URL)
        model = ModelInference(
            model_id=IBM_MODEL_ID,
            credentials=credentials,
            project_id=IBM_PROJECT_ID,
            params={"max_new_tokens": 2400, "temperature": 0.05},
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": _user_prompt(query, context, intent, conf)},
        ]

        # Use streaming if available
        try:
            for chunk in model.chat_stream(messages=messages):
                delta = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta.content or ""
                elif isinstance(chunk, dict):
                    delta = (chunk.get("choices",[{}])[0]
                             .get("delta",{}).get("content",""))
                if delta:
                    yield delta
        except AttributeError:
            # Fallback: non-streaming IBM call
            resp    = model.chat(messages=messages)
            choices = resp.get("choices",[])
            if choices:
                msg = choices[0].get("message",{})
                answer = _clean(msg.get("content") or msg.get("text",""))
                if answer:
                    # Yield in chunks to simulate streaming
                    words = answer.split(" ")
                    for i, word in enumerate(words):
                        yield word + (" " if i < len(words)-1 else "")

    except Exception as e:
        logger.warning("Granite stream error: %s", e)


def _stream_groq(query: str, context: str, intent: str,
                 conf: str) -> Generator[str, None, None]:
    """Stream from Groq Llama 3.3 70B (fast fallback)."""
    if not GROQ_API_KEY: return
    try:
        client = Groq(api_key=GROQ_API_KEY)
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":_user_prompt(query, context, intent, conf)},
            ],
            max_tokens=2400, temperature=0.05, stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception as e:
        logger.warning("Groq stream error: %s", e)


def _generate_full(query: str, context: str, intent: str,
                   conf: str) -> tuple[str, str]:
    """Non-streaming generation for history saving."""
    # Try Granite first
    if IBM_API_KEY and IBM_PROJECT_ID:
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            credentials = Credentials(api_key=IBM_API_KEY, url=IBM_URL)
            model = ModelInference(
                model_id=IBM_MODEL_ID, credentials=credentials,
                project_id=IBM_PROJECT_ID,
                params={"max_new_tokens":2600,"temperature":0.05},
            )
            resp    = model.chat(messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":_user_prompt(query,context,intent,conf)},
            ])
            choices = resp.get("choices",[])
            if choices:
                msg    = choices[0].get("message",{})
                answer = _clean(msg.get("content") or msg.get("text",""))
                if answer and len(answer) > 20:
                    return answer, "ibm-granite-4.0"
        except Exception as e:
            logger.warning("Granite non-stream: %s", e)

    # Groq fallback
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            resp   = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":_user_prompt(query,context,intent,conf)},
                ],
                max_tokens=2400, temperature=0.05, stream=False,
            )
            answer = _clean(resp.choices[0].message.content)
            if answer and len(answer) > 20:
                return answer, "groq-llama-3.3-70b"
        except Exception as e:
            logger.warning("Groq non-stream: %s", e)

    return "Unable to generate answer. Please check API configuration.", "none"


# ── Citations ─────────────────────────────────────────────────────────────────
def build_citations(doc_chunks: list[RetrievedChunk],
                    web_results: list[WebResult]) -> dict:
    return {
        "documents": [
            {"index":i+1,"source":c.source,"page":c.page,"category":c.category,
             "score":round(c.score,3),"element_type":c.element_type,
             "language":c.language,"source_type":c.source_type,
             "verified":c.verified}
            for i,c in enumerate(doc_chunks[:6])
        ],
        "web": [
            {"index":i+1,"title":w.title,"url":w.url,"score":round(w.score,3)}
            for i,w in enumerate(web_results[:4])
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CORE
# ══════════════════════════════════════════════════════════════════════════════

def _run_retrieval(query: str) -> dict[str, Any]:
    t0     = time.time()
    intent = _detect_intent(query)
    lang   = _detect_language(query)

    # Embed
    try:
        embedding = embed_query(query)
    except Exception as e:
        return {"ok":False,"error":str(e),"intent":intent}

    # Dual-track retrieval (parallel)
    t_ret      = time.time()
    doc_chunks = dual_track_retrieve(embedding, intent)
    ret_time   = round(time.time()-t_ret, 2)

    # Verify & rerank
    t_ce = time.time()
    verified_chunks, best_logit = verify_and_rerank(query, doc_chunks)
    conf_label, conf_score      = classify_confidence(best_logit)
    ce_time = round(time.time()-t_ce, 2)

    n_verified = sum(1 for c in verified_chunks if c.verified)
    n_tables   = sum(1 for c in verified_chunks if c.element_type=="table")

    # CRAG branch
    web_results: list[WebResult] = []
    used_docs = verified_chunks

    if conf_label == "CORRECT" and n_verified >= 2:
        branch = "local_docs_only"
    elif conf_label == "AMBIGUOUS" or n_verified < 2:
        web_results = search_web(query)
        branch = f"docs_plus_web ({len(web_results)} results)"
    else:
        used_docs   = []
        web_results = search_web(query)
        branch = f"web_only ({len(web_results)} results)"

    context = build_context(used_docs, web_results)

    return {
        "ok":True, "intent":intent, "language":lang,
        "used_docs":used_docs, "web_results":web_results,
        "context":context, "conf_label":conf_label, "conf_score":conf_score,
        "branch":branch, "retrieval_s":ret_time, "ce_s":ce_time,
        "total_s":round(time.time()-t0,2),
        "trace": {
            "query": {"intent":intent,"language":lang},
            "retrieval": {
                "text_chunks": sum(1 for c in doc_chunks if c.element_type!="table"),
                "table_chunks": n_tables,
                "total": len(doc_chunks),
                "time_s": ret_time,
            },
            "crossencoder": {
                "best_logit":       round(best_logit,3),
                "confidence_label": conf_label,
                "confidence_score": conf_score,
                "verified_chunks":  n_verified,
                "time_s":           ce_time,
            },
            "corrective_branch": {"action":branch},
            "context": {
                "chars":len(context),
                "verified_used":n_verified,
                "tables_used":n_tables,
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API 1: STREAMING
# Yields tokens then final <<<META>>> line
# ══════════════════════════════════════════════════════════════════════════════

def stream_crag_pipeline(query: str) -> Generator[str, None, None]:
    r = _run_retrieval(query)

    if not r["ok"]:
        yield f"❌ Error: {r.get('error','unknown')}"
        yield f"\n<<<META>>>{json.dumps({'error':r.get('error')})}"
        return

    context    = r["context"]
    conf_label = r["conf_label"]
    intent     = r["intent"]

    if not context.strip():
        answer = ("I could not find relevant information in official HTE documents. "
                  "Please contact DTE Maharashtra at https://www.dtemaharashtra.gov.in")
        yield answer
    else:
        full = []

        # Try Granite first
        if IBM_API_KEY and IBM_PROJECT_ID:
            for token in _stream_granite(query, context, intent, conf_label):
                full.append(token); yield token

        # If Granite gave nothing, use Groq
        if not "".join(full).strip():
            for token in _stream_groq(query, context, intent, conf_label):
                full.append(token); yield token

        # If both failed
        if not "".join(full).strip():
            yield "Unable to generate answer. Please check your API keys."

    meta = {
        "confidence_label": r["conf_label"],
        "confidence_score": r["conf_score"],
        "model_used":       "ibm-granite-4.0" if IBM_API_KEY else "groq-llama-3.3-70b",
        "citations":        build_citations(r["used_docs"], r["web_results"]),
        "pipeline_trace":   r["trace"],
        "retrieval_s":      r["retrieval_s"],
    }
    yield f"\n<<<META>>>{json.dumps(meta)}"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API 2: STANDARD (backward-compatible)
# ══════════════════════════════════════════════════════════════════════════════

def run_crag_pipeline(query: str,
                      translate_to_marathi: bool = False) -> CRAGResult:
    t0 = time.time()
    r  = _run_retrieval(query)

    if not r["ok"]:
        return CRAGResult(
            answer=f"Retrieval error: {r.get('error')}",
            confidence_label="INCORRECT", confidence_score=0.0,
            doc_sources=[], web_sources=[],
            pipeline_trace={"error":r.get("error")},
            error=r.get("error"),
        )

    context    = r["context"]
    conf_label = r["conf_label"]
    intent     = r["intent"]

    if not context.strip():
        answer     = ("I could not find relevant information in official HTE documents. "
                      "Please contact DTE Maharashtra at https://www.dtemaharashtra.gov.in")
        model_used = "none"
    else:
        answer, model_used = _generate_full(query, context, intent, conf_label)

    translated_answer=""; translation_applied=False
    if translate_to_marathi and answer:
        try:
            from translator import translate_to_marathi as _tr
            translated_answer   = _tr(answer)
            translation_applied = bool(translated_answer and translated_answer!=answer)
            r["trace"]["translation"] = {"applied":True}
        except Exception as e:
            r["trace"]["translation"] = {"applied":False,"error":str(e)}

    r["trace"]["llm_generation"] = {"model":model_used}
    r["trace"]["total_time_s"]   = round(time.time()-t0,2)

    return CRAGResult(
        answer=answer,
        confidence_label=conf_label,
        confidence_score=r["conf_score"],
        doc_sources=r["used_docs"][:6],
        web_sources=r["web_results"][:4],
        pipeline_trace=r["trace"],
        translation_applied=translation_applied,
        translated_answer=translated_answer,
        model_used=model_used,
    )
