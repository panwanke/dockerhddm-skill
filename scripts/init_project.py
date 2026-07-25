#!/usr/bin/env python3
"""Initialize a reproducible dockerHDDM project from the bundled template.

The project gets a project-local ``env.json`` (generated from
``assets/env.example.json``) that holds the important runtime config: which
dockerHDDM image tag to use, the container mount point, an optional local
source repo, and the default workflow mode. ``env.json`` lives in the project
directory -- never in a global location.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path


TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".gitignore"}

DEFAULT_IMAGE_TAG = "hcp4715/hddm:1.1.0"

AGENTS_MD_REMINDER = (
    "REMINDER: add this line to your AGENTS.md so the agent reads the config "
    "first on every run:\n  每次开始本项目工作前，先阅读项目根目录的 env.json，再动手。"
)


def wanted(relative: Path, mode: str) -> bool:
    parts = relative.parts
    if mode == "inference" and "simulation" in parts:
        return False
    if mode == "simulation" and "inference" in parts:
        return False
    return True


def write_env(target: Path, project_name: str, mode: str, image_tag: str) -> None:
    """Generate <project>/env.json from the bundled template."""
    template = Path(__file__).resolve().parents[1] / "assets" / "env.example.json"
    text = template.read_text(encoding="utf-8")
    text = (
        text.replace("{{PROJECT_NAME}}", project_name)
        .replace("{{WORKFLOW_MODE}}", mode)
        .replace("{{IMAGE_TAG}}", image_tag)
    )
    (target / "env.json").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--mode", choices=("inference", "simulation", "hybrid"), default="hybrid"
    )
    parser.add_argument(
        "--image-tag",
        default=DEFAULT_IMAGE_TAG,
        help="dockerHDDM image tag written into env.json (default: %(default)s). "
        "Other tags (1.0.1, latest, 0.8) are supported but our knowledge is "
        "anchored to 1.1.0.",
    )
    args = parser.parse_args()

    target = args.project_dir.resolve()
    env_path = target / "env.json"
    project_name = args.name or target.name

    # Bootstrap: an existing, non-empty project that only lacks env.json.
    if target.exists() and any(target.iterdir()):
        if env_path.exists():
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {target}")
        target.mkdir(parents=True, exist_ok=True)
        write_env(target, project_name, args.mode, args.image_tag)
        print(f"Bootstrapped {env_path} in existing project {target}")
        print(AGENTS_MD_REMINDER)
        return 0

    target.mkdir(parents=True, exist_ok=True)

    template = Path(__file__).resolve().parents[1] / "assets" / "project-template"
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

    write_env(target, project_name, args.mode, args.image_tag)
    print(f"Initialized {args.mode} dockerHDDM project at {target}")
    print(f"Wrote {env_path} (image_tag={args.image_tag})")
    print(AGENTS_MD_REMINDER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
