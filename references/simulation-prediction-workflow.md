# Simulation and prediction workflow: parameters/theory → data

## Choose the simulation question

| Question | Recommended API |
|---|---|
| fixed standard-DDM parameters | `hddm.generate.gen_rand_data` |
| one LAN/nonstandard parameter vector | `simulator_single_subject` |
| two-stimulus coding design | `simulator_stimcoding` |
| hierarchical conditions/regressions | `simulator_h_c` |
| raw SSM arrays/metadata | `hddm.simulators.simulator` |
| fitted posterior → new data | `model.to_infdata(ppc=True, ...)` |

## Parameter safety

Read `hddm.model_config[model]`:

```python
from hddm.model_config import model_config

spec = model_config["ddm"]
print(spec["params"])
print(spec["param_bounds"])
print(spec["params_default"])
```

For the pinned LAN `ddm`, the source lists parameters `[v, a, z, t]` with
bounds `[-3, .3, .1, .001]` to `[3, 2.5, .9, 2]`. These are training-domain
bounds, not universal scientific priors. Standard HDDM uses a different
boundary convention; the source warns that `a` differs by a factor of two when
comparing standard HDDM and the LAN DDM.

## Fixed-parameter simulation

```python
params = {"v": 1.0, "a": 1.5, "t": 0.30, "z": 0.50}
data, truth = hddm.generate.gen_rand_data(
    params,
    size=200,
    subjs=20,
    seed=2026,
)
```

Save both trial data and `truth`. Validate RT units, response coding, and
per-subject counts exactly as real data.

## Multiple conditions

```python
condition_params = {
    "easy": {"v": 1.5, "a": 1.5, "t": 0.30, "z": 0.50},
    "hard": {"v": 0.5, "a": 1.5, "t": 0.30, "z": 0.50},
}
data, truth = hddm.generate.gen_rand_data(
    condition_params, size=200, subjs=20, seed=2026
)
```

Vary one theoretically targeted parameter first. Use factorial or Latin
hypercube designs when multiple parameters must vary, and record the exact
design table.

## Hierarchical synthetic data

`simulator_h_c` mirrors HDDM model structure and accepts `conditions`,
`depends_on`, regression formulas/covariates, group-only parameters, and fixed
defaults. Use it when the recovery model must match a hierarchical design.

```python
data, truth = hddm.simulators.simulator_h_c(
    n_subjects=40,
    n_trials_per_subject=200,
    model="ddm_hddm_base",
    conditions={"condition": ["easy", "hard"]},
    depends_on={"v": "condition"},
)
```

Inspect its returned truth keys before mapping them to fitted trace names.

## Parameter recovery

For every planned analysis:

1. sample generating parameters across scientifically plausible values;
2. generate many synthetic participants/datasets;
3. fit the exact planned model;
4. compare posterior estimates/intervals with known truth;
5. plot true vs. recovered values and compute bias, RMSE, coverage, and rank;
6. repeat across trial counts, participant counts, and error rates;
7. revise design/model until target effects are recoverable.

Do not assess recovery from a single convenient parameter vector.

## Prediction and visualization

Summarize generated and observed data in the same units:

- choice/error proportions;
- RT density and 0.1/0.5/0.9 quantiles by response;
- condition contrasts;
- individual variability;
- time-on-task effects;
- predicted intervals and empirical coverage.

Save `ground-truth.csv`, `simulation-config.json`, `seed`, raw synthetic trials,
summary tables, and figures. Keep preprocessing separate from plotting.
