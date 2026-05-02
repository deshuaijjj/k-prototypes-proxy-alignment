"""Check whether generated proxy-alignment PDF files contain text/font data."""
from __future__ import annotations

import argparse
from pathlib import Path


def verify_pdf_text(pdf_path: Path) -> bool:
    """Return True if the PDF appears to contain text or font objects."""
    content = pdf_path.read_bytes().decode("latin-1", errors="ignore")
    has_text_operator = "BT" in content and "ET" in content
    has_font_object = "Font" in content or "Type1" in content or "TrueType" in content
    return has_text_operator or has_font_object


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=root / "final_peerj_ai_application_package" / "figures")
    args = parser.parse_args()

    pdf_files = sorted(args.figure_dir.glob("proxy_alignment_*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {args.figure_dir}")
        raise SystemExit(1)

    all_good = True
    for pdf_path in pdf_files:
        ok = verify_pdf_text(pdf_path)
        status = "PASS" if ok else "FAIL (no text/font data detected)"
        print(f"Checking {pdf_path.name}: {status}")
        all_good = all_good and ok

    if not all_good:
        raise SystemExit(1)
    print("All checked PDF figures contain text/font data.")


if __name__ == "__main__":
    main()
