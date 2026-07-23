#!/usr/bin/env python3
"""Initialize a reproducible dockerHDDM project from the bundled template."""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path


TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".gitignore"}


def wanted(relative: Path, mode: str) -> bool:
    parts = relative.parts
    if mode == "inference" and "simulation" in parts:
        return False
    if mode == "simulation" and "inference" in parts:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--mode", choices=("inference", "simulation", "hybrid"), default="hybrid"
    )
    args = parser.parse_args()

    target = args.project_dir.resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {target}")
    target.mkdir(parents=True, exist_ok=True)

    template = Path(__file__).resolve().parents[1] / "assets" / "project-template"
    project_name = args.name or target.name
    replacements = {
        "{{PROJECT_NAME}}": project_name,
        "{{WORKFLOW_MODE}}": args.mode,
        "{{DATE}}": date.today().isoformat(),
    }

    for source in template.rglob("*"):
        relative = source.relative_to(template)
        if not wanted(relative, args.mode):
            continue
        destination = target / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in TEXT_SUFFIXES or source.name == ".gitignore":
            text = source.read_text(encoding="utf-8")
            for old, new in replacements.items():
                text = text.replace(old, new)
            destination.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(source, destination)

    print(f"Initialized {args.mode} dockerHDDM project at {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
