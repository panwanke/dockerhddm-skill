#!/usr/bin/env python3
"""Report dockerHDDM runtime versions and custom API capabilities."""

from __future__ import annotations

import importlib
import inspect
import platform


def version(module) -> str:
    return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    print(f"Python: {platform.python_version()}")
    loaded = {}
    for name in ("numpy", "pandas", "arviz", "pymc", "kabuki", "hddm", "ssms"):
        try:
            loaded[name] = importlib.import_module(name)
            print(f"{name}: {version(loaded[name])}")
        except Exception as error:
            print(f"{name}: IMPORT FAILED: {type(error).__name__}: {error}")

    arviz = loaded.get("arviz")
    if arviz:
        for feature in ("loo", "loo_subsample", "update_subsample", "plot_ppc_dist"):
            print(f"arviz.{feature}: {hasattr(arviz, feature)}")

    kabuki = loaded.get("kabuki")
    if kabuki:
        hierarchical = importlib.import_module("kabuki.hierarchical")
        sample_source = inspect.getsource(hierarchical.Hierarchical.sample)
        for feature in ("chains", "return_infdata", "save_name", "to_infdata"):
            print(f"custom sample supports {feature}: {feature in sample_source}")
        print(
            "n_jobs popped by sample before PyMC2 call:",
            'kwargs.pop("n_jobs"' in sample_source,
        )

    # Version authority check.
    expected_tag = "hcp4715/hddm:1.1.0"
    print(f"\nExpected image tag (this skill's knowledge base): {expected_tag}")
    selected_tag = _read_env_image_tag()
    if selected_tag:
        print(f"env.json image_tag: {selected_tag}")
        if selected_tag != expected_tag:
            print(
                "WARNING: env.json uses a non-1.1.0 tag. Our workflow / API / "
                "debugging guidance is validated against 1.1.0; when behavior "
                "conflicts, 1.1.0 is authoritative."
            )
    else:
        print(
            "NOTE: no project env.json found. Our knowledge is anchored to "
            f"{expected_tag}; other tags may differ."
        )
    return 0


def _read_env_image_tag() -> str | None:
    """Best-effort read of image_tag from a project env.json."""
    candidates = [
        Path.cwd() / "env.json",
        Path("/home/jovyan/work/env.json"),
    ]
    for path in candidates:
        try:
            if path.is_file():
                import json

                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("image_tag")
        except Exception:
            continue
    return None


if __name__ == "__main__":
    raise SystemExit(main())
