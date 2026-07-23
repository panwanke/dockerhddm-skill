from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt


IDATA = Path("models/baseline.nc")
FIGS = Path("figs")
RESULTS = Path("results/diagnostics")
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

idata = az.from_netcdf(IDATA)
summary = az.summary(idata)
summary.to_csv(RESULTS / "posterior-summary.csv")

axes = az.plot_trace(idata, compact=True)
plt.savefig(FIGS / "trace.png", dpi=180, bbox_inches="tight")
plt.close("all")

print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]])
