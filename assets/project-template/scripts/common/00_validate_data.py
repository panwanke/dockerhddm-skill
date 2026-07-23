from pathlib import Path

import pandas as pd


INPUT = Path("data/raw/trials.csv")
OUTPUT = Path("data/processed/trials.csv")
REQUIRED = {"subj_idx", "rt", "response"}


df = pd.read_csv(INPUT)
missing = REQUIRED.difference(df.columns)
assert not missing, f"Missing columns: {sorted(missing)}"
assert df[list(REQUIRED)].notna().all().all(), "Missing required values"
assert (df["rt"] > 0).all(), "RT must be positive"
assert df["rt"].max() < 10, "RT appears to be milliseconds; convert to seconds"
assert set(df["response"].unique()).issubset({0, 1}), "response must be binary"
assert df.groupby("subj_idx").size().min() > 0

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(df.groupby("subj_idx")["rt"].agg(["count", "mean", "min", "max"]))
print(f"Validated {len(df):,} trials -> {OUTPUT}")
