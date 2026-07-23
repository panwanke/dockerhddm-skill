import json
from pathlib import Path

import hddm
import pandas as pd


config = json.loads(Path("config/model.json").read_text(encoding="utf-8"))
data = pd.read_csv("data/processed/simulated-trials.csv")
model_cfg = config["model"]
model = hddm.HDDM(
    data,
    include=model_cfg["include"],
    depends_on=model_cfg.get("depends_on", {}),
    **model_cfg.get("extra_kwargs", {}),
)
sampling = config["sampling"]
idata = model.sample(
    sampling["draws"],
    burn=sampling["burn"],
    chains=sampling["chains"],
    parallel=sampling["parallel"],
    return_infdata=True,
    save_name="models/recovery",
)
print(idata)
