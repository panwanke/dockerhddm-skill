from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IDATA = Path("models/baseline.nc")
FIGS = Path("figs")
RESULTS = Path("results/ppc")
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

idata = az.from_netcdf(IDATA)
assert hasattr(idata, "posterior_predictive"), "Generate PPC in 10_fit.py first"

az.plot_ppc_dist(idata, var_names="rt", num_samples=50)
plt.savefig(FIGS / "ppc-rt-distribution.png", dpi=180, bbox_inches="tight")
plt.close("all")

observed = idata.observed_data.to_dataframe().reset_index()
predicted = idata.posterior_predictive.to_dataframe().reset_index()
quantiles = [0.1, 0.5, 0.9]
table = pd.DataFrame(
    {
        "quantile": quantiles,
        "observed_rt": np.quantile(observed["rt"], quantiles),
        "predicted_rt": np.quantile(predicted["rt"], quantiles),
    }
)
table.to_csv(RESULTS / "rt-quantiles.csv", index=False)
print(table)
