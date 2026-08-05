"""
chroma_loader.py — with detailed debug logging to find exact failure
"""
from __future__ import annotations

import os
import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Always use /tmp on cloud — ./ is read-only on Streamlit Cloud
_raw_path = os.getenv("CHROMA_PATH", "/tmp/chroma_db")
if _raw_path.startswith("./") or _raw_path == "chroma_db":
    CHROMA_PATH = "/tmp/chroma_db"
    logger.warning("Overriding relative CHROMA_PATH to /tmp/chroma_db")
else:
    CHROMA_PATH = _raw_path

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")
HF_TOKEN        = os.getenv("HF_TOKEN", "")

_SENTINEL = os.path.join(CHROMA_PATH, ".hf_downloaded")
_lock     = threading.Lock()


def _log_dir(path: str, label: str = "") -> None:
    p = Path(path)
    prefix = f"[{label}] " if label else ""
    if not p.exists():
        logger.info("%s%s does not exist", prefix, path)
        return
    items = list(p.iterdir())
    logger.info("%s%s — %d items: %s", prefix, path, len(items),
                [i.name for i in items[:15]])


def _is_valid_chroma_db(path: str) -> bool:
    import re
    p = Path(path)
    sqlite = p / "chroma.sqlite3"
    if not sqlite.exists():
        logger.error("FAIL: chroma.sqlite3 missing in %s", path)
        _log_dir(path, "chroma_dir")
        return False
    if sqlite.stat().st_size < 1024:
        logger.error("FAIL: chroma.sqlite3 too small (%d bytes)", sqlite.stat().st_size)
        return False
    uuid_pat = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    uuid_dirs = [d for d in p.iterdir() if d.is_dir() and uuid_pat.match(d.name)]
    if not uuid_dirs:
        logger.error("FAIL: no UUID segment folders in %s", path)
        _log_dir(path, "chroma_dir")
        return False
    for folder in uuid_dirs:
        if any(folder.iterdir()):
            logger.info("VALID: sqlite3=%dKB, %d UUID folders",
                        sqlite.stat().st_size // 1024, len(uuid_dirs))
            return True
    logger.error("FAIL: UUID folders all empty")
    return False


def _try_open_collection(path: str) -> bool:
    try:
        import chromadb
        col_name = os.getenv("CHROMA_COLLECTION", "hte_documents")
        client   = chromadb.PersistentClient(path=path)
        col      = client.get_collection(col_name)
        count    = col.count()
        logger.info("Collection '%s' opened OK — %d chunks", col_name, count)
        return count > 0
    except Exception as exc:
        logger.error("Collection open FAILED: %s", exc)
        return False


def _download_from_hf(target_path: str) -> bool:
    if not HF_DATASET_REPO:
        logger.error("HF_DATASET_REPO is not set")
        return False

    logger.info("HF download start: repo=%s token=%s target=%s",
                HF_DATASET_REPO, "SET" if HF_TOKEN else "NOT SET", target_path)

    tmp_path = target_path + "_tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=tmp_path,
            token=HF_TOKEN or None,
            ignore_patterns=["*.md", "*.txt", ".gitattributes", "README*"],
        )
        logger.info("HF download complete")
    except Exception as exc:
        logger.error("snapshot_download FAILED: %s", exc)
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        return False

    _log_dir(tmp_path, "downloaded")

    # Find chroma.sqlite3 — may be nested
    tmp = Path(tmp_path)
    if not (tmp / "chroma.sqlite3").exists():
        candidates = list(tmp.rglob("chroma.sqlite3"))
        logger.info("sqlite3 not at root — found: %s", candidates)
        if candidates:
            nested = candidates[0].parent
            logger.info("Using nested root: %s", nested)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
            shutil.copytree(str(nested), target_path)
            shutil.rmtree(tmp_path)
        else:
            logger.error("chroma.sqlite3 not found anywhere in download")
            _log_dir(tmp_path, "tmp_full")
            shutil.rmtree(tmp_path)
            return False
    else:
        logger.info("sqlite3 at root — moving to target")
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        shutil.move(tmp_path, target_path)

    _log_dir(target_path, "final")
    return True


def ensure_chroma_downloaded() -> bool:
    logger.info("=== ensure_chroma_downloaded ===")
    logger.info("CHROMA_PATH=%s", CHROMA_PATH)
    logger.info("HF_DATASET_REPO=%s", HF_DATASET_REPO)
    logger.info("HF_TOKEN set=%s", bool(HF_TOKEN))
    logger.info("SENTINEL exists=%s", os.path.isfile(_SENTINEL))

    if os.path.isfile(_SENTINEL):
        logger.info("Sentinel found — validating existing DB")
        if _is_valid_chroma_db(CHROMA_PATH) and _try_open_collection(CHROMA_PATH):
            logger.info("DB valid — ready")
            return True
        logger.warning("Sentinel exists but DB invalid — re-downloading")
        try:
            os.remove(_SENTINEL)
        except Exception:
            pass

    with _lock:
        if os.path.isfile(_SENTINEL):
            return _is_valid_chroma_db(CHROMA_PATH)

        os.makedirs(CHROMA_PATH, exist_ok=True)

        if not _download_from_hf(CHROMA_PATH):
            return False
        if not _is_valid_chroma_db(CHROMA_PATH):
            return False
        if not _try_open_collection(CHROMA_PATH):
            return False

        try:
            Path(_SENTINEL).touch()
            logger.info("Sentinel written")
        except Exception as exc:
            logger.warning("Could not write sentinel: %s", exc)

        logger.info("=== SUCCESS ===")
        return True


def get_chunk_count() -> int:
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        col    = client.get_collection(os.getenv("CHROMA_COLLECTION", "hte_documents"))
        return col.count()
    except Exception:
        return 0
