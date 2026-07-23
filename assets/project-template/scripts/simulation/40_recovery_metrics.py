import re
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


truth = pd.read_csv("results/simulation/ground-truth.csv")
idata = az.from_netcdf("models/recovery.nc")
posterior = az.summary(idata)[["mean", "hdi_3%", "hdi_97%"]].reset_index()
posterior = posterior.rename(columns={posterior.columns[0]: "trace_name"})

pattern = re.compile(r"^(?P<parameter>[A-Za-z0-9]+)_subj[._](?P<subj_idx>.+)$")
rows = []
for row in posterior.itertuples(index=False):
    match = pattern.match(str(row.trace_name))
    if not match:
        continue
    parameter = match.group("parameter")
    if parameter not in truth.columns:
        continue
    subj_text = match.group("subj_idx")
    try:
        subj_idx = int(float(subj_text))
    except ValueError:
        continue
    target = truth.loc[truth["subj_idx"] == subj_idx, parameter]
    if target.empty:
        continue
    rows.append(
        {
            "subj_idx": subj_idx,
            "parameter": parameter,
            "truth": float(target.iloc[0]),
            "estimate": float(row.mean),
            "hdi_low": float(getattr(row, "_2")),
            "hdi_high": float(getattr(row, "_3")),
        }
    )

matched = pd.DataFrame(rows)
output = Path("results/recovery")
output.mkdir(parents=True, exist_ok=True)
posterior.to_csv(output / "posterior-summary.csv", index=False)
if matched.empty:
    raise RuntimeError(
        "No subject-level trace names matched ground truth. Inspect posterior-summary.csv "
        "and adapt the trace-name parser to this model."
    )

matched["error"] = matched["estimate"] - matched["truth"]
matched["covered"] = (
    (matched["truth"] >= matched["hdi_low"])
    & (matched["truth"] <= matched["hdi_high"])
)
metric_rows = []
for parameter, group in matched.groupby("parameter"):
    metric_rows.append(
        {
            "parameter": parameter,
            "bias": group["error"].mean(),
            "rmse": np.sqrt(np.mean(group["error"] ** 2)),
            "coverage": group["covered"].mean(),
            "correlation": group[["truth", "estimate"]].corr().iloc[0, 1],
            "n": len(group),
        }
    )
metrics = pd.DataFrame(metric_rows)
matched.to_csv(output / "matched-estimates.csv", index=False)
metrics.to_csv(output / "metrics.csv", index=False)
print(metrics)
