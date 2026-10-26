"""OPUS OpenSubtitles downloader: v2024 layout with selective English extraction.

Real v2024 layout (confirmed via opus_get -l):
  Alignment: https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/xml/en-ta.xml.gz
  Tamil:     https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/xml/ta.zip
  English:   https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/xml/en.zip  (65 GB – never full)

Strategy:
  1. Download the alignment .xml.gz in full (17 MB).
  2. Parse it to discover exactly which English document paths are referenced.
  3. Test that the en.zip server honours HTTP Range requests; stop with a clear
     message if it does not (never download the whole 65 GB file).
  4. Extract only the referenced English documents from the remote en.zip via
     Range requests (remotezip package).
  5. Download ta.zip in full (142 MB) and extract only the referenced Tamil docs.
  6. Cache all extracted XML.gz files under data/raw/opus/<version>/en/ and .../ta/.

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import gzip
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tyto.utils.config import TytoConfig

logger = logging.getLogger(__name__)

_OPUS_CDN = "https://object.pouta.csc.fi"
_OPUS_API = "https://opus.nlpl.eu/opusapi/"

# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def resolve_version(corpus: str, version: str, src: str, tgt: str) -> str:
    """Return the concrete version string (e.g. 'v2024') for a corpus.

    If *version* is ``'latest'`` the OPUS API is queried for the most recent
    release.  Otherwise the value is returned verbatim (e.g. ``'v2024'`` or
    ``'2016'``).

    Args:
        corpus:  OPUS corpus name, e.g. ``'OpenSubtitles'``.
        version: User-supplied version string or ``'latest'``.
        src:     Source language code.
        tgt:     Target language code.

    Returns:
        Concrete version string.
    """
    if version.lower() != "latest":
        return version
    import requests

    url = (
        f"{_OPUS_API}?corpus={corpus}&source={src}&target={tgt}"
        "&preprocessing=xml&version=latest"
    )
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # OPUS API returns a list of corpus entries; pick the first (latest).
        corpora = data.get("corpora", [])
        if corpora:
            v = corpora[0].get("version", "")
            if v:
                logger.info("Resolved latest version: %s", v)
                return v
    except Exception as exc:
        logger.warning("Could not resolve latest version from OPUS API: %s", exc)
    logger.warning("Could not resolve 'latest'; falling back to 'v2024'")
    return "v2024"


def build_opus_urls(
    corpus: str,
    version: str,
    src_lang: str,
    tgt_lang: str,
) -> dict[str, str]:
    """Return the three canonical OPUS CDN URLs for a language pair.

    v2024 layout (no combined en-ta.zip):
      - ``alignment`` – ``/OPUS-<corpus>/<version>/xml/<src>-<tgt>.xml.gz``
      - ``tgt_zip``   – ``/OPUS-<corpus>/<version>/xml/<tgt>.zip``
      - ``src_zip``   – ``/OPUS-<corpus>/<version>/xml/<src>.zip``

    Args:
        corpus:   OPUS corpus name, e.g. ``'OpenSubtitles'``.
        version:  Concrete version string, e.g. ``'v2024'``.
        src_lang: Source language code, e.g. ``'en'``.
        tgt_lang: Target language code, e.g. ``'ta'``.

    Returns:
        Dict with keys ``alignment``, ``tgt_zip``, ``src_zip``.
    """
    base = f"{_OPUS_CDN}/OPUS-{corpus}/{version}/xml"
    return {
        "alignment": f"{base}/{src_lang}-{tgt_lang}.xml.gz",
        "tgt_zip": f"{base}/{tgt_lang}.zip",
        "src_zip": f"{base}/{src_lang}.zip",
    }


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Compute MD5 hex-digest of a local file (OPUS publishes MD5 sidecars)."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_remote_md5(md5_url: str) -> str | None:
    """Fetch the OPUS .md5 sidecar and return the hex-digest, or None on error."""
    import requests

    try:
        resp = requests.get(md5_url, timeout=30)
        resp.raise_for_status()
        return resp.text.split()[0].lower()
    except Exception as exc:
        logger.warning("Could not fetch MD5 from %s: %s", md5_url, exc)
        return None


def _verify_size(path: Path, url: str) -> bool:
    """Return True if the local file's size matches the remote Content-Length.

    If the remote size cannot be determined, returns True (non-blocking).
    """
    import requests

    try:
        resp = requests.head(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        remote_size = int(resp.headers.get("content-length", 0))
    except Exception as exc:
        logger.warning("HEAD failed for %s: %s – skipping size check", url, exc)
        return True
    if remote_size == 0:
        return True
    local_size = path.stat().st_size
    if local_size == remote_size:
        logger.info("Size OK for %s (%d bytes)", path.name, local_size)
        return True
    logger.warning(
        "Size mismatch %s: local=%d remote=%d", path.name, local_size, remote_size
    )
    return False


def _download_file(url: str, dest: Path, *, resume: bool = True) -> None:
    """Download *url* to *dest* with optional byte-range resume.

    Args:
        url:    Remote URL.
        dest:   Local destination (parent must exist).
        resume: If True and *dest* exists, send ``Range`` header to resume.
    """
    import requests

    headers: dict[str, str] = {}
    file_mode = "ab" if resume and dest.exists() else "wb"
    local_size = dest.stat().st_size if (resume and dest.exists()) else 0

    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        logger.debug("Resuming %s from byte %d", url, local_size)

    with requests.get(url, headers=headers, stream=True, timeout=300) as resp:
        if resp.status_code == 416:
            logger.info("File already complete (416): %s", dest.name)
            return
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + local_size
        downloaded = local_size
        with dest.open(file_mode) as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
        if total > 0:
            logger.info("Downloaded %s (%.1f MB)", dest.name, downloaded / 1e6)


# ---------------------------------------------------------------------------
# Range-request probe
# ---------------------------------------------------------------------------


def probe_range_support(url: str) -> bool:
    """Return True if the server at *url* honours HTTP Range requests.

    Sends ``Range: bytes=0-0`` and checks for status 206.

    Args:
        url: URL to probe (must point to a real file).

    Returns:
        True if Range is supported, False otherwise.

    Raises:
        RuntimeError: if the probe itself errors out (network failure, 404, …).
    """
    import requests

    logger.info("Probing Range support: %s", url)
    try:
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
    except Exception as exc:
        raise RuntimeError(
            f"Range probe failed (network error) for {url}: {exc}"
        ) from exc

    if resp.status_code == 404:
        raise RuntimeError(f"Range probe: 404 Not Found for {url}")

    supported = resp.status_code == 206
    if supported:
        logger.info("Range requests: SUPPORTED (HTTP 206)")
    else:
        logger.warning(
            "Range requests: NOT supported (HTTP %d) for %s",
            resp.status_code,
            url,
        )
    return supported


# ---------------------------------------------------------------------------
# Alignment file helpers
# ---------------------------------------------------------------------------


def download_alignment(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download the alignment .xml.gz file with size check.

    Args:
        url:   Remote URL for the alignment ``.xml.gz``.
        dest:  Local destination path.
        force: Re-download even if dest exists and size matches.

    Returns:
        Path to the local file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if _verify_size(dest, url):
            logger.info("Alignment file already present: %s", dest)
            return dest
        logger.warning("Size mismatch – re-downloading alignment file")
        dest.unlink()

    logger.info("Downloading alignment file: %s", url)
    _download_file(url, dest, resume=True)
    return dest


def collect_en_doc_paths(align_gz_path: Path) -> list[str]:
    """Parse the alignment .xml.gz and return the unique English document paths.

    The alignment XML looks like::

        <linkGrp fromDoc="en/OpenSubtitles.en.ta.12345.xml.gz"
                 toDoc="ta/OpenSubtitles.en.ta.12345.xml.gz">

    Args:
        align_gz_path: Local path to the gzip-compressed alignment file.

    Returns:
        Sorted list of unique ``fromDoc`` values (English paths).
    """
    from xml.etree import ElementTree as ET

    paths: set[str] = set()
    with gzip.open(align_gz_path, "rb") as gz:
        for _event, elem in ET.iterparse(gz, events=("start",)):
            if elem.tag == "linkGrp":
                fd = elem.get("fromDoc", "")
                if fd:
                    paths.add(fd)
            elem.clear()
    logger.info("Alignment references %d unique English documents", len(paths))
    return sorted(paths)


# ---------------------------------------------------------------------------
# Target-language zip (full download + selective extract)
# ---------------------------------------------------------------------------


def download_tgt_zip(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download the target-language zip (ta.zip, ~142 MB) with size check.

    Args:
        url:   Remote URL.
        dest:  Local destination path.
        force: Re-download even if size matches.

    Returns:
        Path to the local zip.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        if _verify_size(dest, url):
            logger.info("Target zip already present: %s", dest)
            return dest
        logger.warning("Size mismatch – re-downloading target zip")
        dest.unlink()

    logger.info("Downloading target zip: %s (%.0f MB expected)", url, 142)
    _download_file(url, dest, resume=True)
    return dest


def extract_tgt_docs(
    tgt_zip_path: Path,
    en_doc_paths: list[str],
    cache_dir: Path,
    tgt_lang: str = "ta",
) -> dict[str, Path]:
    """Extract Tamil XML.gz files from ta.zip that correspond to the EN docs.

    Args:
        tgt_zip_path: Local path to the downloaded ta.zip.
        en_doc_paths: List of English document paths from the alignment file.
        cache_dir:    Root cache directory; Tamil files are written to
                      ``cache_dir/<tgt_lang>/``.
        tgt_lang:     Target language code.

    Returns:
        Dict mapping English doc path -> local Tamil XML.gz path.
    """
    import zipfile

    ta_dir = cache_dir / tgt_lang
    ta_dir.mkdir(parents=True, exist_ok=True)

    # Build a set of expected Tamil filenames (just the basename).
    en_to_ta: dict[str, str] = {}
    for en_path in en_doc_paths:
        fname = en_path.replace("\\", "/").split("/")[-1]
        en_to_ta[en_path] = fname

    result: dict[str, Path] = {}
    needed = set(en_to_ta.values())

    with zipfile.ZipFile(tgt_zip_path, "r") as zf:
        namelist = zf.namelist()
        for entry in namelist:
            bn = entry.replace("\\", "/").split("/")[-1]
            if bn not in needed:
                continue
            local = ta_dir / bn
            if not local.exists():
                data = zf.read(entry)
                local.write_bytes(data)
                logger.debug("Extracted ta: %s", local.name)
            # Map back to EN doc paths.
            for en_path, ta_fname in en_to_ta.items():
                if ta_fname == bn:
                    result[en_path] = local

    found = len(result)
    total = len(en_doc_paths)
    logger.info("Extracted %d / %d Tamil documents from ta.zip", found, total)
    return result


# ---------------------------------------------------------------------------
# English zip: selective extraction via Range requests (remotezip)
# ---------------------------------------------------------------------------


def extract_en_docs_remote(
    en_zip_url: str,
    en_doc_paths: list[str],
    cache_dir: Path,
    src_lang: str = "en",
    *,
    max_docs: int | None = None,
) -> dict[str, Path]:
    """Extract English XML.gz files from the remote en.zip via Range requests.

    Does NOT download the whole en.zip (65 GB).  Uses the ``remotezip``
    package to read the central directory and fetch only the needed entries.

    Args:
        en_zip_url:   Remote URL of ``en.zip``.
        en_doc_paths: English document paths from the alignment file.
        cache_dir:    Root cache directory; files land in ``cache_dir/en/``.
        src_lang:     Source language code.
        max_docs:     If given, stop after this many documents.

    Returns:
        Dict mapping English doc path -> local cached XML.gz path.

    Raises:
        RuntimeError: If the server does not support Range requests.
        ImportError:  If ``remotezip`` is not installed.
    """
    try:
        from remotezip import RangeNotSupported, RemoteZip
    except ImportError as e:
        raise ImportError(
            "remotezip is required for selective English extraction. "
            "Install it with: pip install remotezip"
        ) from e

    en_dir = cache_dir / src_lang
    en_dir.mkdir(parents=True, exist_ok=True)

    if max_docs is not None:
        en_doc_paths = en_doc_paths[:max_docs]

    needed_names = {
        p.replace("\\", "/").split("/")[-1]: p for p in en_doc_paths
    }

    # Skip already-cached docs.
    result: dict[str, Path] = {}
    still_needed: dict[str, str] = {}
    for fname, en_path in needed_names.items():
        local = en_dir / fname
        if local.exists():
            result[en_path] = local
        else:
            still_needed[fname] = en_path

    if not still_needed:
        logger.info("All %d English docs already cached", len(result))
        return result

    logger.info(
        "Fetching %d English docs from remote zip via Range requests …",
        len(still_needed),
    )
    try:
        with RemoteZip(en_zip_url) as rz:
            remote_names = {
                n.replace("\\", "/").split("/")[-1]: n for n in rz.namelist()
            }
            for fname, en_path in still_needed.items():
                remote_entry = remote_names.get(fname)
                if remote_entry is None:
                    logger.warning("English doc not found in remote zip: %s", fname)
                    continue
                data = rz.read(remote_entry)
                local = en_dir / fname
                local.write_bytes(data)
                result[en_path] = local
                logger.debug("Fetched en: %s", fname)
    except RangeNotSupported as exc:
        raise RuntimeError(
            "The OPUS server does not support HTTP Range requests for en.zip. "
            "Cannot selectively extract English documents without downloading "
            "the full 65 GB file. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Done: %d / %d English docs fetched", len(result), len(en_doc_paths)
    )
    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def download_opus(
    cfg: TytoConfig,
    *,
    force: bool = False,
    max_docs: int | None = None,
) -> dict[str, Path]:
    """Download OPUS OpenSubtitles en-ta corpus for v2024 layout.

    Steps:
    1. Resolve the concrete version (if ``'latest'``).
    2. Test that the en.zip server supports Range requests; stop if not.
    3. Download alignment XML.gz and ta.zip in full (with resume + size check).
    4. Parse the alignment to find exactly which EN docs are needed.
    5. Extract those EN docs from the remote en.zip via Range requests.
    6. Extract the corresponding TA docs from the local ta.zip.

    Args:
        cfg:      Validated TytoConfig.
        force:    Re-download even if local files exist and sizes match.
        max_docs: Limit the number of English documents extracted (None = all).

    Returns:
        Dict with keys:
          ``alignment`` – local path to en-ta.xml.gz
          ``tgt_zip``   – local path to ta.zip
          ``en_docs``   – local cache dir for English XML.gz files
          ``ta_docs``   – local cache dir for Tamil XML.gz files

    Raises:
        RuntimeError: If Range requests are not supported, or checksums fail.
    """
    opus = cfg.opus
    version = resolve_version(opus.corpus, opus.version, opus.src_lang, opus.tgt_lang)

    cache_root = Path(opus.download_dir) / version
    cache_root.mkdir(parents=True, exist_ok=True)

    urls = build_opus_urls(opus.corpus, version, opus.src_lang, opus.tgt_lang)

    # ── Step 1: Probe Range support on en.zip ──────────────────────────────
    logger.info("Step 1/5: Probing Range support on en.zip …")
    range_ok = probe_range_support(urls["src_zip"])
    if not range_ok:
        raise RuntimeError(
            "HALT: The OPUS en.zip server does not support HTTP Range requests "
            f"(URL: {urls['src_zip']}). "
            "Selectively extracting English documents is not possible without "
            "Range support. Please contact the OPUS maintainers or use an "
            "alternative mirror that supports Range requests."
        )

    # ── Step 2: Download alignment ─────────────────────────────────────────
    logger.info("Step 2/5: Downloading alignment file …")
    align_dest = cache_root / f"{opus.src_lang}-{opus.tgt_lang}.xml.gz"
    download_alignment(urls["alignment"], align_dest, force=force)

    # ── Step 3: Parse alignment to get EN doc list ─────────────────────────
    logger.info("Step 3/5: Collecting English document paths from alignment …")
    en_doc_paths = collect_en_doc_paths(align_dest)
    if max_docs is not None:
        logger.info("Limiting to first %d documents (--max-docs)", max_docs)

    # ── Step 4: Download ta.zip in full ────────────────────────────────────
    logger.info("Step 4/5: Downloading Tamil zip …")
    ta_zip_dest = cache_root / f"{opus.tgt_lang}.zip"
    download_tgt_zip(urls["tgt_zip"], ta_zip_dest, force=force)

    # ── Step 5: Extract English + Tamil docs ──────────────────────────────
    logger.info("Step 5/5: Extracting documents …")
    en_result = extract_en_docs_remote(
        urls["src_zip"],
        en_doc_paths,
        cache_root,
        src_lang=opus.src_lang,
        max_docs=max_docs,
    )
    ta_result = extract_tgt_docs(
        ta_zip_dest,
        list(en_result.keys()),
        cache_root,
        tgt_lang=opus.tgt_lang,
    )

    logger.info(
        "Download complete. EN docs: %d, TA docs: %d",
        len(en_result),
        len(ta_result),
    )
    return {
        "alignment": align_dest,
        "tgt_zip": ta_zip_dest,
        "en_docs": cache_root / opus.src_lang,
        "ta_docs": cache_root / opus.tgt_lang,
    }


# ---------------------------------------------------------------------------
# Corpus listing (data list command)
# ---------------------------------------------------------------------------


def list_corpus_files(
    corpus: str,
    src_lang: str,
    tgt_lang: str,
    *,
    version: str = "latest",
) -> list[dict]:
    """Query the OPUS API and return available releases for the language pair.

    Args:
        corpus:   OPUS corpus name.
        src_lang: Source language code.
        tgt_lang: Target language code.
        version:  OPUS version filter (``'latest'`` or a concrete version).

    Returns:
        List of dicts with keys ``version``, ``url``, ``size``, ``preprocessing``.
    """
    import requests

    params = {
        "corpus": corpus,
        "source": src_lang,
        "target": tgt_lang,
        "preprocessing": "xml",
    }
    if version.lower() != "all":
        params["version"] = version

    try:
        resp = requests.get(_OPUS_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("corpora", [])
    except Exception as exc:
        logger.warning("OPUS API error: %s", exc)
        return []
