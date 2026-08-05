"""
chroma_loader.py — Downloads ChromaDB from HuggingFace (Xet-compatible)
"""
from __future__ import annotations

import os
import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Streamlit Cloud filesystem is ephemeral — always use /tmp
_raw_path = os.getenv("CHROMA_PATH", "./chroma_db")
CHROMA_PATH = "/tmp/chroma_db" if (_raw_path.startswith("./") or not _raw_path.startswith("/")) else _raw_path

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "sahilsks/hte")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "hte_documents")

_SENTINEL = os.path.join(CHROMA_PATH, ".hf_downloaded")
_lock     = threading.Lock()


def _is_valid() -> bool:
    p = Path(CHROMA_PATH)
    sqlite = p / "chroma.sqlite3"
    if not sqlite.exists() or sqlite.stat().st_size < 1024 * 100:  # must be > 100KB
        logger.error("chroma.sqlite3 missing or too small")
        return False
    import re
    uuid_pat = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    uuid_dirs = [d for d in p.iterdir() if d.is_dir() and uuid_pat.match(d.name)]
    if not uuid_dirs:
        logger.error("No UUID segment folders found")
        return False
    logger.info("DB valid: sqlite3=%dMB, %d UUID dirs",
                sqlite.stat().st_size // (1024*1024), len(uuid_dirs))
    return True


def _list_repo_files() -> list[str]:
    """List all files in the HF dataset repo."""
    from huggingface_hub import list_repo_files
    try:
        files = list(list_repo_files(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN or None,
        ))
        logger.info("Repo files: %s", files)
        return files
    except Exception as exc:
        logger.error("list_repo_files failed: %s", exc)
        return []


def _download_file(repo_path: str, local_path: str) -> bool:
    """Download a single file from HF dataset to local_path."""
    from huggingface_hub import hf_hub_download
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=repo_path,
            repo_type="dataset",
            token=HF_TOKEN or None,
            local_dir=CHROMA_PATH,
            local_dir_use_symlinks=False,   # real files, not symlinks
        )
        size = os.path.getsize(downloaded) if os.path.exists(downloaded) else 0
        logger.info("Downloaded %s → %s (%d KB)", repo_path, downloaded, size // 1024)
        return size > 0
    except Exception as exc:
        logger.error("Download failed for %s: %s", repo_path, exc)
        return False


def _download_all() -> bool:
    """Download every file in the repo, preserving folder structure."""
    files = _list_repo_files()
    if not files:
        return False

    # Filter out non-chroma files
    skip = {".gitattributes", "README.md", ".hf_downloaded"}
    chroma_files = [f for f in files if f not in skip and not f.startswith(".")]

    logger.info("Downloading %d files from sahilsks/hte", len(chroma_files))

    os.makedirs(CHROMA_PATH, exist_ok=True)

    failed = []
    for repo_path in chroma_files:
        local_path = os.path.join(CHROMA_PATH, repo_path)
        ok = _download_file(repo_path, local_path)
        if not ok:
            failed.append(repo_path)

    if failed:
        logger.error("Failed to download: %s", failed)
        # Only fail if sqlite3 itself failed
        if any("chroma.sqlite3" in f for f in failed):
            return False

    return True


def ensure_chroma_downloaded() -> bool:
    logger.info("=== ensure_chroma_downloaded ===")
    logger.info("CHROMA_PATH=%s  HF_REPO=%s  TOKEN=%s",
                CHROMA_PATH, HF_DATASET_REPO, "SET" if HF_TOKEN else "NOT SET")

    # Fast path — sentinel + valid DB
    if os.path.isfile(_SENTINEL) and _is_valid():
        logger.info("Already downloaded and valid — skipping")
        return True

    # Remove stale sentinel
    if os.path.isfile(_SENTINEL):
        os.remove(_SENTINEL)

    with _lock:
        if os.path.isfile(_SENTINEL) and _is_valid():
            return True

        # Clean slate
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        os.makedirs(CHROMA_PATH, exist_ok=True)

        if not _download_all():
            logger.error("Download failed")
            return False

        if not _is_valid():
            logger.error("DB invalid after download")
            return False

        # Verify collection opens
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            col    = client.get_collection(CHROMA_COLLECTION)
            count  = col.count()
            logger.info("Collection '%s' OK — %d chunks", CHROMA_COLLECTION, count)
        except Exception as exc:
            logger.error("Collection open failed: %s", exc)
            return False

        Path(_SENTINEL).touch()
        logger.info("=== SUCCESS — DB ready at %s ===", CHROMA_PATH)
        return True


def get_chunk_count() -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(CHROMA_COLLECTION)
        return col.count()
    except Exception:
        return 0
