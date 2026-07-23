#!/usr/bin/env python3
"""Convert a notebook tree to compact, searchable Markdown references."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REDACTIONS = (
    (re.compile(r"(?i)[A-Z]:[\\/](?:[^\\/\s\"']+[\\/])+"), "<LOCAL_PATH>/"),
    (re.compile(r"(?i)/home/[^/\s\"']+/"), "<HOME>/"),
    (re.compile(r"(?i)C:[\\/]Users[\\/][^\\/\s\"']+[\\/]"), "<HOME>/"),
    (re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Z]{2,}\b"), "<EMAIL>"),
)


def redact(text: str) -> str:
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def source_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def render_output(output: dict) -> str:
    if "text" in output:
        text = output["text"]
        return "".join(text) if isinstance(text, list) else str(text)
    data = output.get("data", {})
    for mime in ("text/plain", "text/markdown"):
        if mime in data:
            value = data[mime]
            return "".join(value) if isinstance(value, list) else str(value)
    return ""


def convert_notebook(path: Path, source_root: Path, include_outputs: bool) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    rel = path.relative_to(source_root).as_posix()
    lines = [
        f"# {path.stem.replace('_', ' ')}",
        "",
        f"> Converted from `{rel}`. Code is preserved; rich outputs are omitted.",
        "",
    ]
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        kind = cell.get("cell_type", "")
        text = redact(source_text(cell)).rstrip()
        if not text:
            continue
        if kind == "markdown":
            lines.extend((text, ""))
        elif kind == "code":
            lines.extend((f"### Code cell {index}", "", "```python", text, "```", ""))
            if include_outputs:
                outputs = [
                    redact(render_output(output)).strip()
                    for output in cell.get("outputs", [])
                ]
                outputs = [output for output in outputs if output]
                if outputs:
                    lines.extend(("Output:", "", "```text", "\n\n".join(outputs), "```", ""))
        elif kind == "raw":
            lines.extend(("```text", text, "```", ""))
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Notebook file or directory")
    parser.add_argument("output", type=Path, help="Destination directory")
    parser.add_argument("--include-outputs", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    source_root = source if source.is_dir() else source.parent
    notebooks = sorted(source.rglob("*.ipynb")) if source.is_dir() else [source]
    args.output.mkdir(parents=True, exist_ok=True)

    index = [
        "# Converted notebook index",
        "",
        "These files preserve narrative and code cells for API lookup. Generated or rich",
        "outputs are omitted by default to keep the skill compact and avoid stale results.",
        "",
        "| Source notebook | Markdown reference |",
        "|---|---|",
    ]
    for notebook in notebooks:
        rel = notebook.relative_to(source_root)
        destination = args.output / rel.with_suffix(".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            convert_notebook(notebook, source_root, args.include_outputs),
            encoding="utf-8",
        )
        index.append(
            f"| `{rel.as_posix()}` | "
            f"[{destination.stem}]({destination.relative_to(args.output).as_posix()}) |"
        )

    (args.output / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Converted {len(notebooks)} notebook(s) into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
