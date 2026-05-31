"""Extract text from downloaded PDFs into per-document .txt files.

Reports words extracted per file so we can spot PDFs that are scanned images
(very low word count) and need OCR or replacement.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_pdfs"
PLANS = DATA / "plans"
SPEECHES = DATA / "speeches"
PLANS.mkdir(parents=True, exist_ok=True)
SPEECHES.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)  # de-hyphenate line breaks
    return text.strip()


def extract(pdf: Path) -> str:
    try:
        reader = PdfReader(str(pdf))
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {pdf.name}: cannot read ({e})")
        return ""
    chunks = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue
    return clean(" ".join(chunks))


def main() -> int:
    report = []
    for pdf in sorted(RAW.glob("*.pdf")):
        text = extract(pdf)
        words = len(text.split())
        if "plan" in pdf.stem:
            out = PLANS / f"{pdf.stem}.txt"
        else:
            out = SPEECHES / f"{pdf.stem}.txt"
        out.write_text(text)
        report.append({"file": pdf.name, "words": words, "out": str(out.relative_to(ROOT))})
        print(f"  {pdf.name}: {words:>6} words")
    (DATA / "extract_report.json").write_text(json.dumps(report, indent=2))
    total = sum(r["words"] for r in report)
    print(f"\nTotal words extracted: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
