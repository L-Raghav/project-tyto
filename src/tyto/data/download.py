"""OPUS OpenSubtitles downloader: resume-safe download with SHA256 checksum verification.

Usage (CLI):
    python -m tyto data download --config configs/default.yaml

Environment:
    PYTHONUTF8=1  (ensures Tamil text survives all file I/O)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tyto.utils.config import TytoConfig

logger = logging.getLogger(__name__)

# OPUS CDN base – files are at:
#   https://object.pouta.csc.fi/OPUS-<corpus>/<version>/xml/<src>-<tgt>.zip   (XML+alignment)
#   https://object.pouta.csc.fi/OPUS-<corpus>/<version>/moses/<src>-<tgt>.txt.zip (Moses)
_OPUS_BASE = "https://object.pouta.csc.fi"


def _expected_sha256_url(zip_url: str) -> str:
    """Return the URL of the MD5/SHA file that OPUS publishes next to each zip."""
    # OPUS publishes <filename>.md5 next to every archive.
    return zip_url + ".md5"


def _sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute SHA-256 hex-digest of a local file in streaming fashion."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute MD5 hex-digest of a local file (OPUS publishes MD5)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_file(url: str, dest: Path, *, resume: bool = True) -> None:
    """Download a URL to *dest* with optional byte-range resume.

    Args:
        url:    Remote URL to fetch.
        dest:   Local destination path (parent directory must exist).
        resume: If True and *dest* exists, send a Range header to resume.
    """
    import requests  # imported here to keep top-level imports light

    headers: dict[str, str] = {}
    file_mode = "ab" if resume and dest.exists() else "wb"
    local_size = dest.stat().st_size if (resume and dest.exists()) else 0

    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        logger.debug("Resuming %s from byte %d", url, local_size)

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        # 416 = range not satisfiable -> file already complete
        if resp.status_code == 416:
            logger.info("File already complete (416): %s", dest.name)
            return
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0)) + local_size
        downloaded = local_size
        with dest.open(file_mode) as fh:
            for chunk_bytes in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk_bytes)
                downloaded += len(chunk_bytes)

        if total > 0:
            logger.info("Downloaded %s  (%.1f MB)", dest.name, downloaded / 1e6)


def _fetch_remote_md5(md5_url: str) -> str | None:
    """Download the OPUS .md5 sidecar and return the hex-digest string, or None."""
    import requests

    try:
        resp = requests.get(md5_url, timeout=30)
        resp.raise_for_status()
        # OPUS format:  <hexdigest>  <filename>
        return resp.text.split()[0].lower()
    except Exception as exc:
        logger.warning("Could not fetch MD5 from %s: %s", md5_url, exc)
        return None


def _verify_or_remove(path: Path, md5_url: str) -> bool:
    """Return True if the local file matches the remote MD5; removes corrupt files.

    If the remote MD5 cannot be fetched we accept the file as-is and return True
    (so the pipeline is not blocked by transient network issues).
    """
    remote_md5 = _fetch_remote_md5(md5_url)
    if remote_md5 is None:
        logger.warning("Skipping checksum verification for %s (MD5 unavailable)", path.name)
        return True
    local_md5 = _md5_of_file(path)
    if local_md5 == remote_md5:
        logger.info("Checksum OK for %s", path.name)
        return True
    logger.error(
        "Checksum MISMATCH for %s  (local=%s  remote=%s) – removing corrupt file",
        path.name,
        local_md5,
        remote_md5,
    )
    path.unlink(missing_ok=True)
    return False


def build_opus_urls(
    corpus: str,
    version: str,
    src_lang: str,
    tgt_lang: str,
) -> dict[str, str]:
    """Build OPUS CDN URLs for the XML+alignment zip.

    The canonical OPUS layout is::

        /OPUS-<corpus>/<version>/xml/<src>-<tgt>.zip

    Args:
        corpus:   OPUS corpus name, e.g. ``"OpenSubtitles"``.
        version:  Release year string, e.g. ``"2016"``.
        src_lang: Source language code, e.g. ``"en"``.
        tgt_lang: Target language code, e.g. ``"ta"``.

    Returns:
        Mapping of label -> URL with keys ``"xml"``.
    """
    base = f"{_OPUS_BASE}/OPUS-{corpus}/{version}"
    pair = f"{src_lang}-{tgt_lang}"
    return {
        "xml": f"{base}/xml/{pair}.zip",
    }


def download_opus(cfg: TytoConfig, *, force: bool = False) -> dict[str, Path]:
    """Download the OPUS OpenSubtitles en-ta corpus, resume-safe with MD5 checksums.

    The function is idempotent: if local files pass their checksum they are
    not re-downloaded.

    Args:
        cfg:   Validated TytoConfig instance.
        force: If True, re-download even if local files exist.

    Returns:
        Mapping of label -> local path for each downloaded archive.
    """
    opus = cfg.opus
    dest_dir = Path(opus.download_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    urls = build_opus_urls(opus.corpus, opus.version, opus.src_lang, opus.tgt_lang)
    local_paths: dict[str, Path] = {}

    for label, url in urls.items():
        filename = url.split("/")[-1]
        local = dest_dir / filename

        if local.exists() and not force:
            logger.info("Found existing archive %s – verifying checksum …", local.name)
            md5_url = url + ".md5"
            ok = _verify_or_remove(local, md5_url)
            if ok:
                local_paths[label] = local
                continue
            # File was corrupt and removed; fall through to re-download.

        logger.info("Downloading %s -> %s", url, local)
        _download_file(url, local, resume=True)

        md5_url = url + ".md5"
        ok = _verify_or_remove(local, md5_url)
        if not ok:
            raise RuntimeError(
                f"Downloaded archive {local.name} failed checksum verification. "
                "Please check network connectivity or OPUS server availability."
            )

        local_paths[label] = local

    logger.info(
        "OPUS download complete. Files: %s",
        {k: str(v) for k, v in local_paths.items()},
    )
    return local_paths


def list_opus_versions(corpus: str = "OpenSubtitles", src_lang: str = "en",
                       tgt_lang: str = "ta") -> None:
    """Print available OPUS versions for the given language pair (informational)."""
    import requests

    pair = f"{src_lang}-{tgt_lang}"
    # Attempt the OPUS API info endpoint.
    url = f"https://opus.nlpl.eu/opusapi/?corpus={corpus}&source={src_lang}&target={tgt_lang}&preprocessing=xml&version=latest"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info("OPUS API response for %s/%s: %s", corpus, pair, data)
        print(data)
    except Exception as exc:
        logger.warning("Could not reach OPUS API: %s", exc)
        print(f"Could not fetch version list: {exc}")
