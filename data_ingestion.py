"""
data_ingestion.py — HTE Multimodal PDF Ingestion Pipeline v4
=============================================================
Supports:
  1. data/**/*.pdf          — existing pipeline (text + tables + OCR)
  2. data/GR/*.pdf          — GR PDFs (same pipeline, source_type=gr_pdf)
  3. data/gitdata/mahGRs/GRs/Higher_and_Technical_Education_Department/*.txt
                            — pre-OCR'd GR text files (en + mr)

New in v4:
  - GR PDF support with language detection + source_type metadata
  - gitdata TXT ingestion with page splitting on "# Page" markers
  - ASCII/markdown table detection in TXT files
  - All new sources feed into the SAME existing ChromaDB schema
  - Zero changes to embedding, chunking, OCR, or ChromaDB schema

Run:
    python data_ingestion.py           # ingest all new files (resumes safely)
    python data_ingestion.py stats     # show DB stats
    python data_ingestion.py verify    # test retrieval
    python data_ingestion.py reset     # wipe and re-ingest everything
"""

from __future__ import annotations

import os

# ── CRITICAL: Clear chromadb env overrides BEFORE any import ─────────────────
for _k in ["CHROMA_API_IMPL", "CHROMA_SERVER_HOST",
           "CHROMA_SERVER_HTTP_PORT", "IS_PERSISTENT"]:
    os.environ.pop(_k, None)
os.environ["ANONYMIZED_TELEMETRY"]               = "False"
os.environ["CHROMA_OTEL_EXPORTER_OTLP_ENDPOINT"] = ""

import glob
import hashlib
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

import chromadb
import pdfplumber
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_NAME    = os.getenv("CHROMA_COLLECTION", "hte_documents")
CHROMA_PATH        = os.getenv("CHROMA_PATH", "./chroma_db")
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
CHUNK_SIZE         = 800
OVERLAP            = 150
OCR_TEXT_THRESHOLD = 50     # chars below which a page is treated as scanned

GITDATA_PATH = (
    "data/gitdata/mahGRs/GRs/Higher_and_Technical_Education_Department"
)
GR_PDF_PATH = "data/GR"

VALID_CATEGORIES = {
    "admission", "fees", "scholarship", "circular", "affiliation",
    "examination", "hostel", "curriculum", "contact", "exams",
    "faculty", "faq", "forms", "notices", "placements", "regulations",
    "syllabus", "gr", "general",
}

# ── OCR (lazy) ────────────────────────────────────────────────────────────────
_ocr_engine = None

def _get_ocr() -> Any:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            use_angle_cls=True, lang="en",
            use_gpu=False, show_log=False, enable_mkldnn=False,
        )
        logger.info("PaddleOCR initialised")
        return _ocr_engine
    except ImportError:
        logger.warning("PaddleOCR not installed — scanned pages skipped")
        return None
    except Exception as exc:
        logger.error("PaddleOCR init failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ── TEXT / OCR EXTRACTION  (unchanged from existing pipeline) ─────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _page_to_image_bytes(page) -> Optional[bytes]:
    try:
        return page.get_pixmap(dpi=200).tobytes("png")
    except Exception as exc:
        logger.warning("Page render failed: %s", exc)
        return None


def _ocr_page(page) -> str:
    ocr = _get_ocr()
    if ocr is None:
        return ""
    img_bytes = _page_to_image_bytes(page)
    if not img_bytes:
        return ""
    try:
        import io
        import numpy as np
        from PIL import Image
        img_array = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        result = ocr.ocr(img_array, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    text, conf = text_info[0], (text_info[1] if len(text_info) > 1 else 1.0)
                    if conf > 0.5 and text.strip():
                        lines.append(text.strip())
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""


def extract_text_from_page(page, page_num: int) -> tuple[str, bool]:
    """Returns (text, was_ocr_used)."""
    try:
        native_text = page.get_text("text").strip()
    except Exception:
        native_text = ""
    if len(native_text) >= OCR_TEXT_THRESHOLD:
        return native_text, False
    ocr_text = _ocr_page(page)
    if ocr_text.strip():
        logger.info("Page %d: OCR extracted %d chars", page_num, len(ocr_text))
        return ocr_text, True
    return native_text, False


def extract_tables_from_page(plumber_page, page_num: int) -> list[str]:
    tables_md = []
    try:
        for table in (plumber_page.extract_tables() or []):
            if not table:
                continue
            md_rows = []
            for row_i, row in enumerate(table):
                cells = [str(c or "").replace("\n", " ").strip() for c in row]
                md_rows.append("| " + " | ".join(cells) + " |")
                if row_i == 0:
                    md_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            md = "\n".join(md_rows)
            if len(md.strip()) > 30:
                tables_md.append(md)
    except Exception as exc:
        logger.warning("Table extraction error page %d: %s", page_num, exc)
    return tables_md


# ══════════════════════════════════════════════════════════════════════════════
# ── PDF PROCESSING  (existing + GR PDFs reuse this) ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(
    pdf_path: str,
    category: str,
    source_type: str = "pdf",
    language: str = "en",
) -> list[dict[str, Any]]:
    """
    Process a PDF — text, OCR, and tables.
    source_type: "pdf" for existing docs, "gr_pdf" for GR folder.
    """
    import fitz
    filename = os.path.basename(pdf_path)
    elements: list[dict[str, Any]] = []
    ocr_pages = native_pages = 0

    try:
        mupdf_doc   = fitz.open(pdf_path)
        plumber_pdf = pdfplumber.open(pdf_path)
    except Exception as exc:
        logger.error("Cannot open PDF %s: %s", filename, exc)
        return []

    try:
        for page_num in range(len(mupdf_doc)):
            mupdf_page = mupdf_doc[page_num]
            real_page  = page_num + 1

            text, used_ocr = extract_text_from_page(mupdf_page, real_page)
            if used_ocr:
                ocr_pages += 1
            else:
                native_pages += 1

            if text.strip():
                elements.append({
                    "text":         text.strip(),
                    "element_type": "ocr_text" if used_ocr else "text",
                    "page":         real_page,
                    "source":       filename,
                    "filename":     filename,
                    "category":     category,
                    "source_type":  source_type,
                    "language":     language,
                    "table_index":  0,
                })

            if not used_ocr and page_num < len(plumber_pdf.pages):
                try:
                    for t_idx, md in enumerate(
                        extract_tables_from_page(plumber_pdf.pages[page_num], real_page)
                    ):
                        elements.append({
                            "text":         md,
                            "element_type": "table",
                            "page":         real_page,
                            "source":       filename,
                            "filename":     filename,
                            "category":     category,
                            "source_type":  source_type,
                            "language":     language,
                            "table_index":  t_idx + 1,
                        })
                except Exception as exc:
                    logger.warning("Table error page %d: %s", real_page, exc)
    finally:
        mupdf_doc.close()
        plumber_pdf.close()

    logger.info(
        "%s: %d elements | %d native | %d OCR",
        filename, len(elements), native_pages, ocr_pages,
    )
    return elements


# ══════════════════════════════════════════════════════════════════════════════
# ── GITDATA TXT PROCESSING  (NEW) ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Matches markdown-style tables: rows starting with | and containing |
_TABLE_ROW_RE  = re.compile(r"^\|.+\|", re.MULTILINE)
# Matches ASCII tables: rows with +---+ or |---|
_ASCII_TABLE_RE = re.compile(r"^\+[-+]+\+", re.MULTILINE)


def _is_table_block(block: str) -> bool:
    """
    Detect if a text block contains a markdown or ASCII table.
    Criteria: at least 3 consecutive lines starting with |
    """
    lines = block.strip().splitlines()
    consecutive = 0
    for line in lines:
        if line.strip().startswith("|"):
            consecutive += 1
            if consecutive >= 3:
                return True
        else:
            consecutive = 0
    # Also check ASCII table format
    if _ASCII_TABLE_RE.search(block):
        return True
    return False


def _detect_language_from_filename(filename: str) -> str:
    """
    Detect language from orgpedia filename convention:
      *.pdf.en.txt → English
      *.pdf.mr.txt → Marathi
    """
    if filename.endswith(".en.txt"):
        return "en"
    if filename.endswith(".mr.txt"):
        return "mr"
    return "en"  # default


def _extract_gr_id(filename: str) -> str:
    """
    Extract GR ID from orgpedia filename.
    Example: 20171012514029708.pdf.en.txt → 20171012514029708
    """
    # Remove .pdf.en.txt or .pdf.mr.txt suffix
    base = filename
    for suffix in [".pdf.en.txt", ".pdf.mr.txt", ".en.txt", ".mr.txt"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def _split_into_pages(text: str) -> list[tuple[int, str]]:
    """
    Split orgpedia TXT content on '# Page N' markers.
    Returns list of (page_number, page_text) tuples.
    If no page markers found, treat entire file as page 1.
    """
    # Match "# Page 1", "# Page 2", etc.
    page_pattern = re.compile(r"^#\s*Page\s+(\d+)", re.IGNORECASE | re.MULTILINE)
    splits = list(page_pattern.finditer(text))

    if not splits:
        return [(1, text.strip())]

    pages = []
    for i, match in enumerate(splits):
        page_num  = int(match.group(1))
        start     = match.end()
        end       = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        page_text = text[start:end].strip()
        if page_text:
            pages.append((page_num, page_text))

    return pages if pages else [(1, text.strip())]


def _split_page_into_blocks(page_text: str) -> list[str]:
    """
    Split a page's text into logical blocks separated by blank lines.
    Keeps table blocks together.
    """
    # Split on 2+ blank lines
    raw_blocks = re.split(r"\n{2,}", page_text.strip())
    blocks = [b.strip() for b in raw_blocks if b.strip()]
    return blocks


def process_gitdata_txt(
    txt_path: str,
    category: str = "gr",
) -> list[dict[str, Any]]:
    """
    Process a pre-OCR'd .txt file from orgpedia/mahGRs.

    Pipeline:
      1. Read text directly (no PDF, no OCR)
      2. Split on "# Page N" markers
      3. Per page: split into blocks → detect tables → classify element_type
      4. Return elements in same format as process_pdf()
    """
    filename = os.path.basename(txt_path)
    language = _detect_language_from_filename(filename)
    gr_id    = _extract_gr_id(filename)
    elements: list[dict[str, Any]] = []

    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    except Exception as exc:
        logger.error("Cannot read %s: %s", filename, exc)
        return []

    if not raw_text.strip():
        return []

    pages = _split_into_pages(raw_text)

    for page_num, page_text in pages:
        if not page_text.strip():
            continue

        blocks = _split_page_into_blocks(page_text)
        table_idx = 0

        for block in blocks:
            if not block.strip():
                continue

            if _is_table_block(block):
                table_idx += 1
                elements.append({
                    "text":         block,
                    "element_type": "table",
                    "page":         page_num,
                    "source":       filename,
                    "filename":     filename,
                    "category":     category,
                    "source_type":  "gitdata",
                    "language":     language,
                    "gr_id":        gr_id,
                    "table_index":  table_idx,
                })
            else:
                elements.append({
                    "text":         block,
                    "element_type": "text",
                    "page":         page_num,
                    "source":       filename,
                    "filename":     filename,
                    "category":     category,
                    "source_type":  "gitdata",
                    "language":     language,
                    "gr_id":        gr_id,
                    "table_index":  0,
                })

    logger.info(
        "%s [%s]: %d elements from %d pages",
        filename, language, len(elements), len(pages),
    )
    return elements


# ══════════════════════════════════════════════════════════════════════════════
# ── GR LANGUAGE DETECTION  (NEW) ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _detect_pdf_language(pdf_path: str) -> str:
    """
    Quick language detection for GR PDFs.
    Checks if filename contains Devanagari characters (Marathi GRs).
    Falls back to checking first page text.
    """
    filename = os.path.basename(pdf_path)

    # Devanagari Unicode block: U+0900–U+097F
    devanagari_in_name = any('\u0900' <= c <= '\u097F' for c in filename)
    if devanagari_in_name:
        return "mr"

    # Sample first page text
    try:
        import fitz
        doc  = fitz.open(pdf_path)
        text = doc[0].get_text("text") if len(doc) > 0 else ""
        doc.close()
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0 and devanagari / total_alpha > 0.3:
            return "mr"
    except Exception:
        pass

    return "en"


# ══════════════════════════════════════════════════════════════════════════════
# ── CHUNKING  (unchanged) ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def chunk_element(
    element: dict[str, Any],
    global_idx: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[dict[str, Any]]:
    """
    Tables → stored whole.
    Text / OCR text → chunked with overlap.
    Extra metadata fields (source_type, language, gr_id) passed through.
    """
    filename     = element["filename"]
    element_type = element["element_type"]
    text         = element["text"]
    page         = element["page"]
    category     = element["category"]
    table_index  = element.get("table_index", 0)
    source_type  = element.get("source_type", "pdf")
    language     = element.get("language", "en")
    gr_id        = element.get("gr_id", "")
    chunks: list[dict[str, Any]] = []

    def _base_meta(chunk_id: str, wi: int = 0, we: int = 0) -> dict:
        return {
            "source":       element["source"],
            "filename":     filename,
            "category":     category,
            "page":         page,
            "chunk_id":     chunk_id,
            "element_type": element_type,
            "table_index":  table_index,
            "source_type":  source_type,
            "language":     language,
            "gr_id":        gr_id,
            "word_start":   wi,
            "word_end":     we,
        }

    if element_type == "table":
        if len(text.strip()) >= 30:
            cid = (f"{hashlib.sha256(filename.encode()).hexdigest()[:8]}"
                   f"_t{table_index}_p{page}")
            chunks.append({
                "text":        text,
                "chunk_index": global_idx,
                **_base_meta(cid, 0, len(text.split())),
            })
    else:
        words, i, local_idx = text.split(), 0, 0
        while i < len(words):
            chunk_words = words[i: i + chunk_size]
            chunk_str   = " ".join(chunk_words)
            if len(chunk_str.strip()) >= 80:
                cid = (f"{hashlib.sha256(filename.encode()).hexdigest()[:8]}"
                       f"_{global_idx + local_idx}")
                chunks.append({
                    "text":        chunk_str,
                    "chunk_index": global_idx + local_idx,
                    **_base_meta(cid, i, i + len(chunk_words)),
                    "page": page if page > 0 else local_idx + 1,
                })
                local_idx += 1
            i += chunk_size - overlap

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# ── EMBEDDING  (REST API — no SDK) ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> Optional[list[float]]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-embedding-001:embedContent?key={GOOGLE_API_KEY}"
    )
    payload = json.dumps({
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text[:8000]}]},
        "taskType": "RETRIEVAL_DOCUMENT",
    }).encode("utf-8")

    for attempt in range(4):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())["embedding"]["values"]
        except Exception as exc:
            err = str(exc)
            if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                wait = 65 * (attempt + 1)
                print(f"\n⚠️  Rate limit — waiting {wait}s (attempt {attempt+1}/4)…")
                time.sleep(wait)
            else:
                print(f"  Embedding error (attempt {attempt+1}): {exc}")
                time.sleep(3)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ── CHROMADB ──────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        col = client.get_collection(name=COLLECTION_NAME)
        logger.info("Resuming — collection has %d chunks", col.count())
    except Exception:
        col = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Created collection: %s", COLLECTION_NAME)
    return col


def _embed_and_store(
    collection: chromadb.Collection,
    all_chunks: list[dict],
    doc_id: int,
    filename: str,
) -> tuple[int, int]:
    """
    Embed and store chunks. Returns (doc_id_after, success_count).
    """
    success = 0
    for chunk in all_chunks:
        embed_text = chunk["text"]
        if chunk["element_type"] == "table":
            embed_text = (
                f"[TABLE from {chunk['filename']} "
                f"category:{chunk['category']} "
                f"lang:{chunk.get('language','en')}]\n{chunk['text']}"
            )

        embedding = get_embedding(embed_text)
        if embedding is None:
            print(f"\n🛑 Embedding failed after {success} chunks. Run again to resume.\n")
            return doc_id, -1   # signal caller to stop

        meta = {
            "source":       chunk["source"],
            "filename":     chunk["filename"],
            "category":     chunk["category"],
            "page":         chunk["page"],
            "chunk_id":     chunk["chunk_id"],
            "chunk_index":  chunk["chunk_index"],
            "element_type": chunk["element_type"],
            "table_index":  chunk.get("table_index", 0),
            "source_type":  chunk.get("source_type", "pdf"),
            "language":     chunk.get("language", "en"),
            "gr_id":        chunk.get("gr_id", ""),
            "word_start":   chunk.get("word_start", 0),
            "word_end":     chunk.get("word_end", 0),
        }

        try:
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[meta],
            )
            doc_id  += 1
            success += 1
            time.sleep(1.2)
        except Exception as exc:
            print(f"  ❌ ChromaDB error: {exc}")
            continue

    return doc_id, success


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN INGESTION ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def ingest_all():
    collection   = get_or_create_collection()
    existing     = collection.get(include=["metadatas"])
    already_done = {m.get("filename", "") for m in existing["metadatas"]}
    print(f"📋 Already embedded: {len(already_done)} files\n")

    # ── Collect all files ─────────────────────────────────────────────────────
    # 1. Existing PDF folders (everything except GR and gitdata)
    existing_pdfs = [
        p for p in sorted(glob.glob("data/**/*.pdf", recursive=True))
        if not p.replace("\\", "/").startswith("data/GR/")
        and "gitdata" not in p
    ]

    # 2. GR PDFs
    gr_pdfs = sorted(glob.glob(f"{GR_PDF_PATH}/**/*.pdf", recursive=True))

    # 3. Gitdata TXT files (en + mr)
    gitdata_txts = sorted(glob.glob(f"{GITDATA_PATH}/**/*.txt", recursive=True))

    total_pdfs = len(existing_pdfs) + len(gr_pdfs)
    print(f"📁 Found {len(existing_pdfs)} existing PDFs | "
          f"{len(gr_pdfs)} GR PDFs | "
          f"{len(gitdata_txts)} gitdata TXT files\n")

    doc_id     = collection.count()
    stats      = {"new_files": 0, "table": 0, "text": 0, "ocr_text": 0}

    # ── 1. Existing PDFs ──────────────────────────────────────────────────────
    for pdf_path in existing_pdfs:
        filename = os.path.basename(pdf_path)
        if filename in already_done:
            print(f"⏭️  Skipping (done): {filename}")
            continue

        raw_cat  = os.path.basename(os.path.dirname(pdf_path)).lower()
        category = raw_cat if raw_cat in VALID_CATEGORIES else "general"

        print(f"\n📄 [EXISTING PDF] {filename} [{category}]")
        stats["new_files"] += 1
        elements = process_pdf(pdf_path, category, source_type="pdf")
        if not elements:
            print("  ⚠️  No content — skipping")
            continue

        all_chunks = _build_chunks(elements, doc_id)
        _log_chunks(all_chunks, stats)
        doc_id, success = _embed_and_store(collection, all_chunks, doc_id, filename)
        if success == -1:
            return
        print(f"  ✅ {success}/{len(all_chunks)} chunks")

    # ── 2. GR PDFs ────────────────────────────────────────────────────────────
    for pdf_path in gr_pdfs:
        filename = os.path.basename(pdf_path)
        if filename in already_done:
            print(f"⏭️  Skipping (done): {filename}")
            continue

        language = _detect_pdf_language(pdf_path)
        print(f"\n📄 [GR PDF | {language}] {filename}")
        stats["new_files"] += 1
        elements = process_pdf(
            pdf_path, category="gr",
            source_type="gr_pdf", language=language,
        )
        if not elements:
            print("  ⚠️  No content — skipping")
            continue

        all_chunks = _build_chunks(elements, doc_id)
        _log_chunks(all_chunks, stats)
        doc_id, success = _embed_and_store(collection, all_chunks, doc_id, filename)
        if success == -1:
            return
        print(f"  ✅ {success}/{len(all_chunks)} chunks")

    # ── 3. Gitdata TXT files ──────────────────────────────────────────────────
    for txt_path in gitdata_txts:
        filename = os.path.basename(txt_path)
        if filename in already_done:
            print(f"⏭️  Skipping (done): {filename}")
            continue

        lang = _detect_language_from_filename(filename)
        print(f"\n📝 [GITDATA TXT | {lang}] {filename}")
        stats["new_files"] += 1
        elements = process_gitdata_txt(txt_path, category="gr")
        if not elements:
            print("  ⚠️  No content — skipping")
            continue

        all_chunks = _build_chunks(elements, doc_id)
        _log_chunks(all_chunks, stats)
        doc_id, success = _embed_and_store(collection, all_chunks, doc_id, filename)
        if success == -1:
            return
        print(f"  ✅ {success}/{len(all_chunks)} chunks")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"✅ Ingestion complete!")
    print(f"   Total chunks : {doc_id}")
    print(f"   New files    : {stats['new_files']}")
    print(f"   Table chunks : {stats['table']}")
    print(f"   Text chunks  : {stats['text']}")
    print(f"   OCR chunks   : {stats['ocr_text']}")
    print(f"{'='*60}\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_chunks(elements: list[dict], doc_id: int) -> list[dict]:
    all_chunks: list[dict] = []
    for element in elements:
        all_chunks.extend(chunk_element(element, global_idx=doc_id + len(all_chunks)))
    return all_chunks


def _log_chunks(all_chunks: list[dict], stats: dict) -> None:
    t  = sum(1 for c in all_chunks if c["element_type"] == "table")
    tx = sum(1 for c in all_chunks if c["element_type"] == "text")
    oc = sum(1 for c in all_chunks if c["element_type"] == "ocr_text")
    stats["table"]    += t
    stats["text"]     += tx
    stats["ocr_text"] += oc
    print(f"  📦 {len(all_chunks)} chunks → {t} table | {tx} text | {oc} OCR")


# ══════════════════════════════════════════════════════════════════════════════
# ── UTILITIES ─────────────────────────────────────════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════

def show_stats():
    collection = get_or_create_collection()
    existing   = collection.get(include=["metadatas"])
    sources, counts, source_types, languages = {}, {}, {}, {}

    for meta in existing["metadatas"]:
        src = meta.get("filename", "unknown")
        cat = meta.get("category", "unknown")
        et  = meta.get("element_type", "text")
        st  = meta.get("source_type", "pdf")
        lg  = meta.get("language", "en")
        key = f"[{cat}] {src}"
        sources[key]      = sources.get(key, 0) + 1
        counts[et]        = counts.get(et, 0) + 1
        source_types[st]  = source_types.get(st, 0) + 1
        languages[lg]     = languages.get(lg, 0) + 1

    print(f"\n📊 ChromaDB Stats — {COLLECTION_NAME}")
    print(f"   Total chunks  : {collection.count()}")
    print(f"\n   By element type:")
    for et, n in sorted(counts.items()):
        print(f"     {et:12s} : {n}")
    print(f"\n   By source type:")
    for st, n in sorted(source_types.items()):
        print(f"     {st:12s} : {n}")
    print(f"\n   By language:")
    for lg, n in sorted(languages.items()):
        print(f"     {lg:4s}         : {n}")
    print(f"\n   Documents     : {len(sources)}")
    for src, count in sorted(sources.items())[:20]:   # show top 20
        print(f"   {count:4d} chunks │ {src}")
    if len(sources) > 20:
        print(f"   ... and {len(sources)-20} more files")
    print()


def verify_alignment():
    print("\n🔍 Verifying retrieval…")
    collection = get_or_create_collection()
    if collection.count() == 0:
        print("⚠️  Collection empty.")
        return

    test_queries = [
        "fee structure engineering Maharashtra",
        "scholarship economically backward class",
        "government resolution higher technical education",
    ]
    for query in test_queries:
        embedding = get_embedding(query)
        if not embedding:
            continue
        results = collection.query(
            query_embeddings=[embedding], n_results=2,
            include=["documents", "metadatas", "distances"],
        )
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        print(f"\n   Query: '{query}'")
        for doc, meta, dist in zip(docs, metas, dists):
            print(f"   [{meta.get('element_type','?').upper()}]"
                  f"[{meta.get('source_type','?')}]"
                  f"[{meta.get('language','?')}]"
                  f" {meta.get('source','?')}  score={round(1-dist,3)}")
            print(f"   → {doc[:80].replace(chr(10),' ')}…")
    print()


def reset_and_reingest():
    confirm = input("⚠️  Delete all chunks and re-ingest everything? Type 'yes': ")
    if confirm.strip().lower() != "yes":
        print("Cancelled.")
        return
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️  Deleted '{COLLECTION_NAME}'")
    except Exception:
        print("No existing collection to delete.")
    ingest_all()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ingest"
    if   cmd == "stats":   show_stats()
    elif cmd == "verify":  verify_alignment()
    elif cmd == "reset":   reset_and_reingest()
    else:                  ingest_all(); show_stats()