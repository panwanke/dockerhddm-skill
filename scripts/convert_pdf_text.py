#!/usr/bin/env python3
"""Convert a text PDF to page-delimited UTF-8 Markdown with PyMuPDF."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import fitz

    pdf = args.pdf.resolve()
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        f"# {pdf.stem.replace('_', ' ')}",
        "",
        f"> Text-layer conversion from `{pdf.name}`. Verify tables and equations against",
        "> the PDF before using exact values.",
        "",
    ]
    with fitz.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            sections.extend((f"## Page {page_number}", "", text, ""))

    args.output.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(f"Converted {pdf.name} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
