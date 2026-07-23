import json
from pathlib import Path

import arviz as az


config = json.loads(Path("config/model.json").read_text(encoding="utf-8"))
comparison = config["comparison"]
models = {
    label: az.from_netcdf(path)
    for label, path in comparison["models"].items()
}
if len(models) < 2:
    raise ValueError(
        "Add at least two fitted .nc artifacts to comparison.models in config/model.json"
    )

result = az.compare(models, ic=comparison["method"], var_name="log_lik")
output = Path("results/comparison")
output.mkdir(parents=True, exist_ok=True)
result.to_csv(output / f"{comparison['method']}-comparison.csv")
print(result)
