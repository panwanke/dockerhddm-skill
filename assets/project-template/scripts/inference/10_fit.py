import json
from pathlib import Path

import hddm
import pandas as pd


config = json.loads(Path("config/model.json").read_text(encoding="utf-8"))
df = pd.read_csv(config["data_file"])
assert df["rt"].max() < 10, "RT must be in seconds"

model_cfg = config["model"]
kwargs = {
    "include": model_cfg["include"],
    "depends_on": model_cfg.get("depends_on", {}),
    **model_cfg.get("extra_kwargs", {}),
}
model_class = model_cfg["class"]
if model_class == "HDDM":
    model = hddm.HDDM(df, **kwargs)
elif model_class == "HDDMRegressor":
    model = hddm.HDDMRegressor(df, model_cfg["formula"], **kwargs)
elif model_class == "HDDMStimCoding":
    model = hddm.HDDMStimCoding(df, **kwargs)
else:
    raise ValueError(f"Unsupported model class: {model_class}")

sampling = config["sampling"]
model.sample(
    sampling["draws"],
    burn=sampling["burn"],
    chains=sampling["chains"],
    parallel=sampling["parallel"],
    return_infdata=True,
    save_name=sampling["save_name"],
)

# Keep n_jobs out of sample(): the custom wrapper forwards unknown keywords to
# PyMC2. Generate expensive groups in this explicit second stage.
idata_cfg = config["inference_data"]
idata = model.to_infdata(
    sample_prior=idata_cfg["sample_prior"],
    ppc=idata_cfg["ppc"],
    n_ppc=idata_cfg["n_ppc"],
    loglike=idata_cfg["loglike"],
    n_loglike=idata_cfg["n_loglike"],
    parallel=True,
    n_jobs=idata_cfg["n_jobs"],
    save_name=sampling["save_name"],
)
print(idata)
