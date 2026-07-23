#!/usr/bin/env python3
"""Estimate lower-bound array memory for dockerHDDM sampling products."""

from __future__ import annotations

import argparse


def gib(value: float) -> str:
    return f"{value / 1024**3:.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, required=True)
    parser.add_argument("--trials-per-subject", type=int, required=True)
    parser.add_argument("--parameters", type=int, required=True)
    parser.add_argument("--draws", type=int, required=True, help="Retained draws per chain")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--n-ppc", type=int, default=0)
    parser.add_argument("--bytes-per-value", type=int, default=8)
    parser.add_argument("--merge-factor", type=float, default=4.0)
    args = parser.parse_args()

    observations = args.subjects * args.trials_per_subject
    trace = args.parameters * args.draws * args.chains * args.bytes_per_value
    loglike = observations * args.draws * args.chains * args.bytes_per_value
    ppc = observations * args.n_ppc * args.chains * 2 * args.bytes_per_value
    total = trace + loglike + ppc

    print(f"Observations: {observations:,}")
    print(f"Trace arrays (lower bound): {gib(trace)}")
    print(f"Pointwise log-likelihood (lower bound): {gib(loglike)}")
    print(f"PPC RT+response arrays (lower bound): {gib(ppc)}")
    print(f"Array subtotal: {gib(total)}")
    print(f"Conservative merge peak ({args.merge_factor:g}x): {gib(total * args.merge_factor)}")
    print("This excludes Python objects, model copies, pandas/xarray indexes, and OS overhead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
