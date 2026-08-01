"""
crawler.py — Smart HTE Document Crawler
========================================
Responsible ONLY for:
  1. Crawling official HTE websites
  2. Downloading new PDFs / HTML pages
  3. Passing each PDF to data_ingestion.py for extraction + embedding
  4. Tracking ingested URLs in Supabase / local JSON

All extraction, chunking, embedding, and ChromaDB storage is
delegated to data_ingestion.py — single pipeline for both
manual and automatic ingestion.

Usage
-----
python crawler.py --run-now        # crawl all seeds
python crawler.py --status         # show ingestion log
python crawler.py --url <url>      # ingest single URL
python crawler.py --test           # test 1 seed (5 pages)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH       = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hte_documents")
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_ANON_KEY", "")

CRAWL_INTERVAL_HOURS = int(os.getenv("CRAWL_INTERVAL_HOURS", "24"))
MAX_PAGES_PER_SEED   = int(os.getenv("MAX_PAGES_PER_DOMAIN", "60"))
REQUEST_DELAY        = 1.5   # seconds between requests — be polite
REQUEST_TIMEOUT      = 25
MAX_PDF_SIZE_MB      = 25

# Browser-like headers — bypasses User-Agent blocking on most govt sites
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/pdf,*/*;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection":      "keep-alive",
}

# ── Seed URLs ─────────────────────────────────────────────────────────────────
# Targeted deep links — not just root domains.
# Category = ChromaDB category tag for retrieved chunks.

SEED_URLS: list[dict[str, str]] = [
    # DTE Maharashtra
    {
        "url":      "https://dtemaharashtra.gov.in/",
        "category": "general",
        "label":    "DTE Maharashtra Home",
    },
    {
        "url":      "https://dtemaharashtra.gov.in/FeeRegulation/StaticPages/HomePage.aspx",
        "category": "fees",
        "label":    "DTE Fee Regulation",
    },
    {
        "url":      "https://dtemaharashtra.gov.in/Notifications",
        "category": "circular",
        "label":    "DTE Notifications",
    },
    # CET Cell
    {
        "url":      "https://cetcell.mahacet.org/",
        "category": "admission",
        "label":    "CET Cell Home",
    },
    {
        "url":      "https://cetcell.mahacet.org/notification/",
        "category": "admission",
        "label":    "CET Cell Notifications",
    },
    # Maharashtra GRs
    {
        "url":      "https://gr.maharashtra.gov.in/Site/Home/Index.aspx",
        "category": "circular",
        "label":    "Maharashtra GRs",
    },
    # AICTE
    {
        "url":      "https://www.aicte-india.org/approval/2024-25",
        "category": "affiliation",
        "label":    "AICTE Approval 2024-25",
    },
    # UGC
    {
        "url":      "https://www.ugc.gov.in/page/Scholarships-and-Fellowships.aspx",
        "category": "scholarship",
        "label":    "UGC Scholarships",
    },
    # MSBTE
    {
        "url":      "https://msbte.org.in/",
        "category": "examination",
        "label":    "MSBTE Home",
    },
    {
        "url":      "https://msbte.org.in/portal/circulars/",
        "category": "circular",
        "label":    "MSBTE Circulars",
    },
    # MahaDBT (public scholarship portal)
    {
        "url":      "https://mahadbt.maharashtra.gov.in/",
        "category": "scholarship",
        "label":    "MahaDBT Scholarships",
    },
    # Education dept GRs
    {
        "url":      "https://www.maharashtra.gov.in/Site/upload/Government%20Resolutions/English/",
        "category": "circular",
        "label":    "Maharashtra English GRs",
    },
]

# Allowed domains — crawler will NOT follow links outside these
ALLOWED_DOMAINS = {
    "dtemaharashtra.gov.in",
    "dte.maharashtra.gov.in",
    "cetcell.mahacet.org",
    "mahacet.org",
    "gr.maharashtra.gov.in",
    "maharashtra.gov.in",
    "aicte-india.org",
    "ugc.gov.in",
    "ugc.ac.in",
    "msbte.org.in",
    "mahadbt.maharashtra.gov.in",
    "education.gov.in",
    "tec.gov.in",
}

# Keywords that signal a high-priority link
PRIORITY_KEYWORDS = [
    "fee", "circular", "notice", "scholarship", "admission",
    "gr", "notification", "pdf", "brochure", "schedule",
    "cap", "mht", "cet", "affiliation", "regulation",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    url: str
    status: str        # "ingested" | "skipped" | "failed"
    chunks_added: int = 0
    error: Optional[str] = None


# ── Ingestion Log (Supabase → local JSON fallback) ────────────────────────────

class IngestionLog:
    TABLE = "crawler_log"

    def __init__(self):
        self._supabase   = None
        self._local: dict[str, dict] = {}
        self._local_path = "./crawler_log.json"
        self._init()

    def _init(self):
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                from supabase import create_client
                self._supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                logger.info("IngestionLog: Supabase backend")
            except Exception as exc:
                logger.warning("Supabase unavailable → local JSON: %s", exc)
        if not self._supabase:
            self._load_local()

    def _load_local(self):
        import json
        if os.path.exists(self._local_path):
            try:
                with open(self._local_path) as f:
                    self._local = json.load(f)
            except Exception:
                self._local = {}

    def _save_local(self):
        import json
        with open(self._local_path, "w") as f:
            json.dump(self._local, f, indent=2)

    def is_seen(self, url: str, content_hash: str) -> bool:
        if self._supabase:
            try:
                rows = (
                    self._supabase.table(self.TABLE)
                    .select("content_hash")
                    .eq("url", url)
                    .execute()
                ).data or []
                return any(r.get("content_hash") == content_hash for r in rows)
            except Exception as exc:
                logger.warning("IngestionLog.is_seen error: %s", exc)
        return self._local.get(url, {}).get("content_hash") == content_hash

    def mark_done(self, url: str, content_hash: str, chunk_count: int, status: str = "ingested"):
        record = {
            "url":          url,
            "content_hash": content_hash,
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
            "chunk_count":  chunk_count,
            "status":       status,
        }
        if self._supabase:
            try:
                self._supabase.table(self.TABLE).upsert(record).execute()
                return
            except Exception as exc:
                logger.warning("IngestionLog.mark_done error: %s", exc)
        self._local[url] = record
        self._save_local()

    def get_stats(self) -> dict[str, Any]:
        if self._supabase:
            try:
                rows = self._supabase.table(self.TABLE).select("*").execute().data or []
                return {
                    "total_docs":    len(rows),
                    "total_chunks":  sum(r.get("chunk_count", 0) for r in rows),
                    "last_ingested": max((r.get("ingested_at", "") for r in rows), default="never"),
                    "backend":       "supabase",
                }
            except Exception:
                pass
        return {
            "total_docs":    len(self._local),
            "total_chunks":  sum(v.get("chunk_count", 0) for v in self._local.values()),
            "last_ingested": max(
                (v.get("ingested_at", "") for v in self._local.values()), default="never"
            ),
            "backend": "local_json",
        }


_ingestion_log = IngestionLog()


# ── HTTP client ───────────────────────────────────────────────────────────────

_http: Optional[httpx.Client] = None

def _get_http() -> httpx.Client:
    global _http
    if _http is None:
        _http = httpx.Client(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=BROWSER_HEADERS,
        )
    return _http


def _content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _is_allowed(url: str) -> bool:
    try:
        domain = urlparse(url).netloc.lstrip("www.")
        return any(domain == d or domain.endswith("." + d) for d in ALLOWED_DOMAINS)
    except Exception:
        return False


def _is_priority(url: str) -> bool:
    return any(kw in url.lower() for kw in PRIORITY_KEYWORDS)


# ── Core: ingest a single PDF via data_ingestion.py ──────────────────────────

def ingest_pdf_bytes(
    pdf_bytes: bytes,
    source_url: str,
    category: str,
    filename: str,
) -> int:
    """
    Delegates ALL extraction + embedding + storage to data_ingestion.py.
    Returns number of chunks added.
    """
    # Import data_ingestion functions — single pipeline
    try:
        from data_ingestion import (
            extract_with_docling,
            extract_with_pdfplumber,
            chunk_element,
            get_embedding,
            DOCLING_AVAILABLE,
            COLLECTION_NAME,
            CHROMA_PATH as DI_CHROMA_PATH,
        )
        import chromadb as _chromadb

        # Save bytes to a temp file (Docling / pdfplumber need a path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            # Extract using same pipeline as manual ingestion
            if DOCLING_AVAILABLE:
                elements = extract_with_docling(tmp_path, category)
                if not elements:
                    elements = extract_with_pdfplumber(tmp_path, category)
            else:
                elements = extract_with_pdfplumber(tmp_path, category)

            # Override source with URL (not temp path)
            for el in elements:
                el["source"]   = filename
                el["filename"] = filename

        finally:
            os.unlink(tmp_path)

        if not elements:
            logger.warning("No content extracted from %s", source_url)
            return 0

        # Chunk
        all_chunks = []
        for element in elements:
            chunks = chunk_element(element, global_idx=len(all_chunks))
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # Embed + store
        client     = _chromadb.PersistentClient(path=DI_CHROMA_PATH)
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Get current max doc_id
        doc_id = collection.count()
        success = 0

        for chunk in all_chunks:
            embed_text = chunk["text"]
            if chunk["element_type"] == "table":
                embed_text = (
                    f"[TABLE from {chunk['filename']} "
                    f"category:{chunk['category']}]\n{chunk['text']}"
                )

            embedding = get_embedding(embed_text)
            if embedding is None:
                logger.error("Rate limit during crawler ingestion — stopping this PDF")
                break

            try:
                collection.add(
                    ids=[f"doc_{doc_id}"],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[{
                        "source":       chunk["source"],
                        "filename":     chunk["filename"],
                        "category":     chunk["category"],
                        "page":         chunk["page"],
                        "chunk_id":     chunk["chunk_id"],
                        "chunk_index":  chunk["chunk_index"],
                        "element_type": chunk["element_type"],
                        "table_index":  chunk.get("table_index", 0),
                        "word_start":   chunk.get("word_start", 0),
                        "word_end":     chunk.get("word_end", 0),
                        "crawled_from": source_url,
                    }],
                )
                doc_id  += 1
                success += 1
                time.sleep(1.2)   # respect Gemini free tier
            except Exception as e:
                logger.error("ChromaDB add failed: %s", e)

        tables = sum(1 for c in all_chunks if c["element_type"] == "table")
        logger.info(
            "Ingested %s → %d/%d chunks (%d tables)",
            filename, success, len(all_chunks), tables,
        )
        return success

    except ImportError as e:
        logger.error("data_ingestion.py import failed: %s", e)
        return 0


def ingest_html_text(
    text: str,
    source_url: str,
    category: str,
) -> int:
    """
    Ingests plain HTML text using data_ingestion's chunker + embedder.
    Used for HTML pages (not PDFs).
    """
    try:
        from data_ingestion import (
            chunk_element,
            get_embedding,
            COLLECTION_NAME,
            CHROMA_PATH as DI_CHROMA_PATH,
        )
        import chromadb as _chromadb
        from urllib.parse import urlparse as _up

        filename = _up(source_url).path.split("/")[-1] or "page"
        element  = {
            "text":         text,
            "element_type": "text",
            "page":         0,
            "source":       filename,
            "filename":     filename,
            "category":     category,
            "table_index":  0,
        }
        all_chunks = chunk_element(element, global_idx=0)
        if not all_chunks:
            return 0

        client = _chromadb.PersistentClient(path=DI_CHROMA_PATH)
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            collection = client.create_collection(COLLECTION_NAME)

        doc_id  = collection.count()
        success = 0
        for chunk in all_chunks:
            embedding = get_embedding(chunk["text"])
            if embedding is None:
                break
            try:
                collection.add(
                    ids=[f"doc_{doc_id}"],
                    embeddings=[embedding],
                    documents=[chunk["text"]],
                    metadatas=[{
                        **{
                            "source":       chunk["source"],
                            "filename":     chunk["filename"],
                            "category":     chunk["category"],
                            "page":         chunk["page"],
                            "chunk_id":     chunk["chunk_id"],
                            "chunk_index":  chunk["chunk_index"],
                            "element_type": "text",
                            "table_index":  0,
                            "word_start":   chunk.get("word_start", 0),
                            "word_end":     chunk.get("word_end", 0),
                            "crawled_from": source_url,
                        }
                    }],
                )
                doc_id  += 1
                success += 1
                time.sleep(1.2)
            except Exception as e:
                logger.error("ChromaDB add failed: %s", e)
        return success

    except ImportError as e:
        logger.error("data_ingestion.py import failed: %s", e)
        return 0


# ── Main fetch + ingest ───────────────────────────────────────────────────────

def fetch_and_ingest(
    url: str,
    category: str,
    force: bool = False,
) -> IngestionResult:
    http = _get_http()

    # Size check via HEAD
    try:
        head   = http.head(url)
        c_type = head.headers.get("content-type", "").lower()
        c_len  = int(head.headers.get("content-length", 0))
        if c_len > MAX_PDF_SIZE_MB * 1024 * 1024:
            return IngestionResult(url=url, status="skipped",
                                   error=f"Too large: {c_len/1e6:.1f}MB")
    except Exception:
        c_type = ""

    # Fetch
    try:
        resp = http.get(url)
        resp.raise_for_status()
    except Exception as exc:
        return IngestionResult(url=url, status="failed", error=str(exc))

    raw        = resp.content
    actual_ct  = resp.headers.get("content-type", "").lower()
    c_hash     = _content_hash(raw)

    if not force and _ingestion_log.is_seen(url, c_hash):
        return IngestionResult(url=url, status="skipped")

    is_pdf = "pdf" in actual_ct or url.lower().endswith(".pdf")

    if is_pdf:
        filename = urlparse(url).path.split("/")[-1] or "document.pdf"
        added    = ingest_pdf_bytes(raw, url, category, filename)
    else:
        # HTML — extract clean text
        soup  = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        if len(text.strip()) < 150:
            _ingestion_log.mark_done(url, c_hash, 0, "skipped_empty")
            return IngestionResult(url=url, status="skipped", error="Too little text")
        added = ingest_html_text(text, url, category)

    if added > 0:
        _ingestion_log.mark_done(url, c_hash, added, "ingested")
        return IngestionResult(url=url, status="ingested", chunks_added=added)
    else:
        _ingestion_log.mark_done(url, c_hash, 0, "failed")
        return IngestionResult(url=url, status="failed", error="0 chunks produced")


# ── BFS Crawler ───────────────────────────────────────────────────────────────

def crawl_seed(
    seed: dict[str, str],
    max_pages: int = MAX_PAGES_PER_SEED,
) -> list[IngestionResult]:
    url      = seed["url"]
    category = seed["category"]
    label    = seed.get("label", url)
    http     = _get_http()

    results: list[IngestionResult] = []
    queue:   list[str] = [url]
    visited: set[str]  = set()

    logger.info("▶ Crawling: %s  [%s]", label, category)

    while queue and len(visited) < max_pages:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        time.sleep(REQUEST_DELAY)

        # ── Direct PDF ────────────────────────────────────────────────────────
        if current.lower().endswith(".pdf"):
            logger.info("  PDF found: %s", current)
            result = fetch_and_ingest(current, category)
            results.append(result)
            if result.status == "ingested":
                logger.info("  ✓ %d chunks  ← %s", result.chunks_added, current)
            elif result.status == "failed":
                logger.warning("  ✗ failed: %s  (%s)", current, result.error)
            continue

        # ── HTML page ─────────────────────────────────────────────────────────
        try:
            resp = http.get(current)
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.warning("  Fetch failed: %s  (%s)", current, exc)
            results.append(IngestionResult(url=current, status="failed", error=str(exc)))
            continue

        # Parse + queue links
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        for a in soup.find_all("a", href=True):
            href = urljoin(current, a["href"])
            if not _is_allowed(href) or href in visited or href in queue:
                continue
            if _is_priority(href):
                queue.insert(0, href)   # priority → front
            else:
                queue.append(href)

        # Ingest this HTML page itself
        text = re.sub(r"\n{3,}", "\n\n",
                      soup.get_text(separator="\n", strip=True))
        if len(text.strip()) > 200:
            c_hash = _content_hash(html)
            if not _ingestion_log.is_seen(current, c_hash):
                added = ingest_html_text(text, current, category)
                if added > 0:
                    _ingestion_log.mark_done(current, c_hash, added, "ingested")
                    results.append(IngestionResult(url=current, status="ingested", chunks_added=added))
                    logger.info("  ✓ HTML %d chunks ← %s", added, current)
            else:
                results.append(IngestionResult(url=current, status="skipped"))

    logger.info("  Seed done — %d URLs visited", len(visited))
    return results


# ── Full crawl ────────────────────────────────────────────────────────────────

def run_full_crawl(force: bool = False) -> dict[str, Any]:
    started   = datetime.now(timezone.utc).isoformat()
    logger.info("════ Full crawl started: %s ════", started)
    t0        = time.time()
    all_res: list[IngestionResult] = []

    for seed in SEED_URLS:
        try:
            all_res.extend(crawl_seed(seed))
        except Exception as exc:
            logger.error("Seed %s failed: %s", seed["url"], exc)

    ingested = [r for r in all_res if r.status == "ingested"]
    summary  = {
        "started_at":   started,
        "finished_at":  datetime.now(timezone.utc).isoformat(),
        "duration_s":   round(time.time() - t0, 1),
        "total_urls":   len(all_res),
        "ingested":     len(ingested),
        "skipped":      sum(1 for r in all_res if r.status == "skipped"),
        "failed":       sum(1 for r in all_res if r.status == "failed"),
        "total_chunks": sum(r.chunks_added for r in ingested),
    }
    logger.info("════ Crawl complete: %s ════", summary)
    return summary


# ── APScheduler ──────────────────────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None

def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    _scheduler.add_job(
        run_full_crawl,
        trigger="interval",
        hours=CRAWL_INTERVAL_HOURS,
        id="full_crawl",
        name="HTE Document Crawler",
        next_run_time=None,
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    logger.info("Crawler scheduler started — every %dh IST", CRAWL_INTERVAL_HOURS)
    return _scheduler


def stop_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)


def get_next_crawl_time() -> Optional[str]:
    if _scheduler and _scheduler.running:
        job = _scheduler.get_job("full_crawl")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
    return None


def trigger_crawl_now() -> dict[str, Any]:
    if _scheduler and _scheduler.running:
        _scheduler.modify_job("full_crawl", next_run_time=datetime.now(timezone.utc))
        return {"status": "triggered", "message": "Crawl triggered"}
    return run_full_crawl()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTE Smart Crawler")
    parser.add_argument("--run-now",  action="store_true", help="Full crawl now")
    parser.add_argument("--test",     action="store_true", help="Test: 1 seed, 5 pages")
    parser.add_argument("--status",   action="store_true", help="Show ingestion log stats")
    parser.add_argument("--url",      type=str,            help="Ingest a single URL")
    parser.add_argument("--category", type=str, default="general")
    parser.add_argument("--force",    action="store_true", help="Re-ingest even if seen")
    args = parser.parse_args()

    if args.status:
        stats = _ingestion_log.get_stats()
        print("\n── Ingestion Log ─────────────────────────")
        for k, v in stats.items():
            print(f"  {k:20s}: {v}")
        print()

    elif args.url:
        r = fetch_and_ingest(args.url, args.category, force=args.force)
        print(f"\n  URL    : {r.url}")
        print(f"  Status : {r.status}")
        print(f"  Chunks : {r.chunks_added}")
        if r.error:
            print(f"  Error  : {r.error}")
        print()

    elif args.test:
        print("\n── Test crawl (1 seed, 5 pages) ──────────")
        results = crawl_seed(SEED_URLS[0], max_pages=5)
        for r in results:
            print(f"  {r.status:10s} {r.chunks_added:4d} chunks  {r.url[:70]}")
        print()

    elif args.run_now:
        summary = run_full_crawl(force=args.force)
        print("\n── Crawl Summary ─────────────────────────")
        for k, v in summary.items():
            print(f"  {k:20s}: {v}")
        print()

    else:
        parser.print_help()