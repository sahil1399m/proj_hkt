"""
chroma_loader.py — Downloads ChromaDB from HuggingFace Hub on first cold start,
                   validates the structure, then serves from local disk.

KEY FIXES vs original:
  1. Validates DB after download (checks chroma.sqlite3 + at least one UUID folder)
  2. Sentinel is only written AFTER validation passes
  3. Re-downloads if sentinel exists but DB is corrupted
  4. Forces crag.py to re-init ChromaDB AFTER download completes
  5. Handles flat vs nested HF download structures
"""
from __future__ import annotations

import os
import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")

_SENTINEL = os.path.join(CHROMA_PATH, ".hf_downloaded")
_lock     = threading.Lock()


# ── Validation ────────────────────────────────────────────────────────────────

def _is_valid_chroma_db(path: str) -> bool:
    """
    ChromaDB PersistentClient requires:
      1. chroma.sqlite3 — the metadata/index file
      2. At least one UUID-named subfolder containing segment files
    """
    p = Path(path)

    # Check sqlite3 exists and is non-empty
    sqlite = p / "chroma.sqlite3"
    if not sqlite.exists() or sqlite.stat().st_size < 1024:
        logger.warning("chroma.sqlite3 missing or too small: %s", sqlite)
        return False

    # Check for UUID segment folders
    import re
    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    uuid_folders = [
        d for d in p.iterdir()
        if d.is_dir() and uuid_pattern.match(d.name)
    ]
    if not uuid_folders:
        logger.warning("No UUID segment folders found in %s", path)
        return False

    # Check at least one segment folder has data
    for folder in uuid_folders:
        if any(folder.iterdir()):
            logger.info(
                "Valid ChromaDB found: sqlite3=%dKB, %d segment folders",
                sqlite.stat().st_size // 1024,
                len(uuid_folders),
            )
            return True

    logger.warning("UUID folders exist but are all empty")
    return False


def _try_open_collection(path: str) -> bool:
    """Actually open ChromaDB and count docs — ultimate validation."""
    try:
        import chromadb
        collection_name = os.getenv("CHROMA_COLLECTION", "hte_documents")
        client = chromadb.PersistentClient(path=path)
        col    = client.get_collection(collection_name)
        count  = col.count()
        logger.info("ChromaDB opened OK — %d chunks", count)
        return count > 0
    except Exception as exc:
        logger.error("ChromaDB open failed: %s", exc)
        return False


# ── Download ──────────────────────────────────────────────────────────────────

def _download_from_hf(target_path: str) -> bool:
    """
    Downloads the dataset from HuggingFace.
    Handles both flat and nested structures.
    """
    if not HF_DATASET_REPO:
        logger.error("HF_DATASET_REPO not set in environment/secrets")
        return False

    # Use a temp dir so partial downloads don't corrupt the target
    tmp_path = target_path + "_tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    logger.info("Downloading ChromaDB from HF: %s → %s", HF_DATASET_REPO, tmp_path)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=tmp_path,
            token=HF_TOKEN or None,
            ignore_patterns=["*.md", "*.txt", ".gitattributes", "README*"],
        )
    except Exception as exc:
        logger.error("HuggingFace download failed: %s", exc)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        return False

    # ── Structure detection ───────────────────────────────────────────────
    # HF sometimes nests files under a subfolder matching the repo name
    # e.g. tmp_path/chroma_db/chroma.sqlite3  instead of  tmp_path/chroma.sqlite3
    tmp = Path(tmp_path)
    sqlite_direct = tmp / "chroma.sqlite3"

    if not sqlite_direct.exists():
        # Look one level deeper
        candidates = list(tmp.rglob("chroma.sqlite3"))
        if candidates:
            # Use the parent of the first found sqlite3
            nested_root = candidates[0].parent
            logger.info("Nested structure detected — using: %s", nested_root)
            # Move nested root to target
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(str(nested_root), target_path)
            shutil.rmtree(tmp_path)
        else:
            logger.error("chroma.sqlite3 not found anywhere in download")
            shutil.rmtree(tmp_path)
            return False
    else:
        # Flat structure — move directly
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(tmp_path, target_path)

    return True


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_chroma_downloaded() -> bool:
    """
    Call once at startup (before crag.py initialises ChromaDB).
    Blocks until the DB is ready and validated.
    Returns True if DB is ready, False on failure.
    """
    # Fast path — sentinel exists AND DB is still valid
    if os.path.isfile(_SENTINEL):
        if _is_valid_chroma_db(CHROMA_PATH) and _try_open_collection(CHROMA_PATH):
            logger.info("ChromaDB already on disk and valid — skipping download")
            return True
        else:
            logger.warning("Sentinel exists but DB is invalid — re-downloading")
            os.remove(_SENTINEL)

    with _lock:
        # Double-check inside lock
        if os.path.isfile(_SENTINEL):
            return _is_valid_chroma_db(CHROMA_PATH)

        # Download
        if not _download_from_hf(CHROMA_PATH):
            return False

        # Validate
        if not _is_valid_chroma_db(CHROMA_PATH):
            logger.error("Downloaded DB failed structure validation")
            return False

        if not _try_open_collection(CHROMA_PATH):
            logger.error("Downloaded DB failed open/count validation")
            return False

        # Only write sentinel after full validation
        Path(_SENTINEL).touch()
        logger.info("ChromaDB ready at %s", CHROMA_PATH)
        return True


def get_chunk_count() -> int:
    """Quick count for the sidebar — returns 0 on any error."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(os.getenv("CHROMA_COLLECTION", "hte_documents"))
        return col.count()
    except Exception:
        return 0
