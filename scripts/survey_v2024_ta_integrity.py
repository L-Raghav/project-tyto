"""Standalone integrity survey for OPUS OpenSubtitles v2024 Tamil zip.

Checks whether Tamil dependent vowel signs (U+0BBE-U+0BCC) are present at
healthy rates (~0.22-0.26) or severely missing across documents.
Uses only Python stdlib + zipfile (no torch or third-party packages).
"""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path

V2024_TA_ZIP = Path("data/raw/opus/v2024/ta.zip")
MAX_LINES_PER_DOC = 10_000
VOWEL_SIGN_START = 0x0BBE
VOWEL_SIGN_END = 0x0BCC


def is_tamil_vowel_sign(ch: str) -> bool:
    code = ord(ch)
    return VOWEL_SIGN_START <= code <= VOWEL_SIGN_END


def is_tamil_char(ch: str) -> bool:
    return 0x0B80 <= ord(ch) <= 0x0BFF


def vowel_sign_share(lines: list[str]) -> tuple[float, int, int]:
    """Return (vowel_sign_share, vowel_sign_count, tamil_char_count)."""
    vowel_count = 0
    tamil_count = 0
    for line in lines:
        for ch in line:
            if is_tamil_char(ch):
                tamil_count += 1
                if is_tamil_vowel_sign(ch):
                    vowel_count += 1
    share = (vowel_count / tamil_count) if tamil_count > 0 else 0.0
    return share, vowel_count, tamil_count


def main() -> None:
    if not V2024_TA_ZIP.exists():
        print("data/raw/opus/v2024/ta.zip not found — skipping survey")
        return

    doc_lines: dict[str, list[str]] = defaultdict(list)

    with zipfile.ZipFile(V2024_TA_ZIP, "r") as zf:
        ta_members = [
            name for name in zf.namelist()
            if name.endswith(".ta") or name.endswith(".txt") or "/ta/" in name
        ]
        if not ta_members:
            ta_members = [name for name in zf.namelist() if not name.endswith("/")]

        for member in ta_members:
            doc_key = Path(member).parent.as_posix() or member
            with zf.open(member, "r") as fh:
                for raw_line in fh:
                    if len(doc_lines[doc_key]) >= MAX_LINES_PER_DOC:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if line:
                        doc_lines[doc_key].append(line)

    if not doc_lines:
        print("No text lines found inside data/raw/opus/v2024/ta.zip")
        return

    total_docs = len(doc_lines)
    deficient_docs = 0
    shares: list[float] = []

    print(f"Surveyed {total_docs} document(s) (up to {MAX_LINES_PER_DOC:,} lines each):")
    for doc_key, lines in sorted(doc_lines.items()):
        share, v_cnt, t_cnt = vowel_sign_share(lines)
        shares.append(share)
        if share < 0.10:
            deficient_docs += 1
        print(
            f"  {doc_key}: vowel_share={share:.4f} "
            f"({v_cnt}/{t_cnt} Tamil chars, {len(lines)} lines)"
        )

    avg_share = sum(shares) / len(shares)
    pct_deficient = (deficient_docs / total_docs) * 100.0
    print("\nSummary:")
    print(f"  Documents surveyed : {total_docs}")
    print(f"  Deficient (<0.10)  : {deficient_docs} ({pct_deficient:.1f}%)")
    print(f"  Mean vowel share   : {avg_share:.4f} (healthy ~0.22-0.26)")


if __name__ == "__main__":
    main()
