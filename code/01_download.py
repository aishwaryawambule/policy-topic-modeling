"""Download Nepal policy PDFs from a list of verified URLs.

Skips downloads that already exist and are valid PDFs.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import urllib.request
import ssl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "sources.json"
RAW = DATA / "raw_pdfs"
RAW.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.6 Safari/605.1.15"
)
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def is_pdf(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 5000:
        return False
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


def fetch(url: str, dest: Path, timeout: int = 120) -> tuple[bool, str]:
    if is_pdf(dest):
        return True, f"cached ({dest.stat().st_size//1024} KB)"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            data = r.read()
        dest.write_bytes(data)
        if is_pdf(dest):
            return True, f"downloaded ({len(data)//1024} KB)"
        return False, f"not a PDF (got {len(data)} bytes)"
    except Exception as e:  # noqa: BLE001
        return False, f"error: {e.__class__.__name__}: {e}"


def main() -> int:
    sources = json.loads(SRC.read_text())
    summary: list[dict] = []
    for category in ("plans", "budget_speeches"):
        for item in sources[category]:
            label = item["label"]
            url = item["url"]
            dest = RAW / f"{label}.pdf"
            ok, msg = fetch(url, dest)
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {category}/{label}: {msg}")
            summary.append({"category": category, "label": label, "ok": ok, "msg": msg})
            time.sleep(0.5)
    n_ok = sum(1 for s in summary if s["ok"])
    print(f"\n{n_ok}/{len(summary)} files OK.")
    (DATA / "download_report.json").write_text(json.dumps(summary, indent=2))
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
