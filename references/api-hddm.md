# HDDM API map

## Data and model construction

`hddm.HDDM(data, ...)` expects one row per trial with `rt` in seconds,
`response`, and `subj_idx`. `depends_on` creates separate parameter families
for unique values of one or more data columns.

`hddm.HDDMRegressor(data, models, ...)` uses Patsy formulas such as
`"v ~ 1 + C(condition)"`. `group_only_regressors=True` avoids subject-level
regression slopes. `keep_regressor_trace=True` enables some PPCs at a large
storage cost.

`hddm.HDDMStimCoding` models physical stimulus/choice bias. It requires a
two-level `stim_col`; `split_param="z"` maps one stimulus to `z` and the other
to `1-z`, while `"v"` changes the drift sign. The pinned source contains a fix
that appends `z` without replacing an existing include list.

## Classic generation

`hddm.generate.gen_rand_params(include=(), cond_dict=None, seed=None)` returns
valid random DDM parameters. With `cond_dict`, it returns condition-specific and
merged parameter maps.

`hddm.generate.gen_rand_data(params=None, ..., **kwargs)` wraps Kabuki
generation, validates parameter bounds, supports subjects/conditions, adds
optional fast/slow outliers, and returns synthetic trials plus subject truth.

`hddm.generate.add_outliers` modifies generated data with explicit fast/slow
contaminants. Use it for robustness tests, not routine data cleaning.

## Simulator family

- `simulator(**kwargs)`: raw RT, choice, metadata arrays through `ssms`;
- `simulator_single_subject(...)`: HDDM-ready data for one parameter vector;
- `simulator_stimcoding(...)`: two-stimulus synthetic design;
- `simulator_h_c(...)`: hierarchical conditions/regressions and truth map.

The basic wrapper aliases model name `weibull` to `weibull_cdf`.

## Fit artifacts

The customized `sample` wrapper can create:

- `.db`: PyMC2 trace;
- `.hddm`: pickled HDDM object;
- `.nc`: ArviZ/xarray inference artifact.

Loading `.nc` with `az.from_netcdf` is normally faster and safer for downstream
analysis than unpickling the full model. Keep the full model when later API
calls require its node graph.

## Parameter meaning

- `v`: mean direction/rate of evidence accumulation;
- `a`: boundary separation/response caution;
- `z`: relative starting point/bias;
- `t`: nondecision time;
- `sv`, `sz`, `st`: across-trial variability in drift, start, and nondecision;
- model-specific parameters: read `model_config[model]`.

Meaning depends on response/stimulus coding and the model architecture. A
negative `v` is not automatically an error.
