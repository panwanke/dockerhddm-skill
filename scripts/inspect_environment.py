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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
