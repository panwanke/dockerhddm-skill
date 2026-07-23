import json
from pathlib import Path

import hddm
import pandas as pd


config = json.loads(Path("config/model.json").read_text(encoding="utf-8"))
sim = config["simulation"]
data, subject_parameters = hddm.generate.gen_rand_data(
    sim["parameters"],
    size=sim["trials_per_subject"],
    subjs=sim["subjects"],
    seed=sim["seed"],
)
data["simulation_seed"] = sim["seed"]

output = Path("data/processed/simulated-trials.csv")
truth = Path("results/simulation/ground-truth.csv")
output.parent.mkdir(parents=True, exist_ok=True)
truth.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output, index=False)

if isinstance(subject_parameters, list):
    truth_df = pd.DataFrame(subject_parameters)
    truth_df.insert(0, "subj_idx", range(len(truth_df)))
elif isinstance(subject_parameters, dict) and subject_parameters:
    first = next(iter(subject_parameters.values()))
    if isinstance(first, list):
        rows = []
        for condition, values in subject_parameters.items():
            for subj_idx, params in enumerate(values):
                rows.append({"subj_idx": subj_idx, "condition": condition, **params})
        truth_df = pd.DataFrame(rows)
    elif isinstance(first, dict):
        truth_df = pd.DataFrame(
            [
                {"subj_idx": 0, "condition": condition, **params}
                for condition, params in subject_parameters.items()
            ]
        )
    else:
        truth_df = pd.DataFrame([{"subj_idx": 0, **subject_parameters}])
else:
    raise TypeError(f"Unexpected truth structure: {type(subject_parameters)}")
truth_df.to_csv(truth, index=False)
print(data.groupby("subj_idx")["rt"].agg(["count", "mean", "min", "max"]))
