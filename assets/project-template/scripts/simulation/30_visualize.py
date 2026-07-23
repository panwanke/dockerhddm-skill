from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data = pd.read_csv("data/processed/simulated-trials.csv")
figs = Path("figs")
results = Path("results/simulation")
figs.mkdir(parents=True, exist_ok=True)
results.mkdir(parents=True, exist_ok=True)

summary = (
    data.groupby("subj_idx")
    .agg(n=("rt", "size"), mean_rt=("rt", "mean"), choice_rate=("response", "mean"))
    .reset_index()
)
summary.to_csv(results / "behavior-summary.csv", index=False)

sns.histplot(data=data, x="rt", hue="response", element="step", stat="density")
plt.savefig(figs / "simulated-rt-by-response.png", dpi=180, bbox_inches="tight")
plt.close("all")
print(summary.describe())
