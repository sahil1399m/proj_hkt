"""
chroma_loader.py — Downloads ChromaDB from HuggingFace Hub on first cold start,
                   then serves from local disk on every subsequent run.
"""
import os
import logging
import threading
from pathlib import Path
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")
CHROMA_PATH     = os.getenv("CHROMA_PATH", "./chroma_db")

# Sentinel — if this file exists, the DB is already on disk
_SENTINEL = os.path.join(CHROMA_PATH, ".hf_downloaded")

_lock = threading.Lock()


def _already_downloaded() -> bool:
    return os.path.isfile(_SENTINEL)


def ensure_chroma_downloaded() -> bool:
    """
    Call once at startup. Blocks until done.
    Returns True if DB is ready, False on failure.
    """
    if _already_downloaded():
        logger.info("ChromaDB already on disk — skipping download")
        return True

    with _lock:
        if _already_downloaded():          # double-check inside lock
            return True

        if not HF_DATASET_REPO:
            logger.error("HF_DATASET_REPO not set")
            return False

        logger.info("Downloading ChromaDB from %s …", HF_DATASET_REPO)
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                local_dir=CHROMA_PATH,
                token=HF_TOKEN or None,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )
            Path(_SENTINEL).touch()
            logger.info("ChromaDB downloaded to %s", CHROMA_PATH)
            return True
        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return False