# lan visualizations

> Converted from `hddm/hddm/examples/lan_visualizations.ipynb`. Code is preserved; rich outputs are omitted.

## New Visualizations

### Code cell 2

```python
import hddm
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
```

### Generate some Data

### Code cell 4

```python
# Metadata
nmcmc = 2000
model = "angle"
n_samples = 1000
includes = hddm.simulators.model_config[model]["hddm_include"]
```

### Code cell 5

```python
data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(
    n_subjects=5,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions=None,
    depends_on=None,
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 6

```python
data
```

### Code cell 7

```python
# Define the HDDM model
hddmnn_model = hddm.HDDMnn(
    data, informative=False, include=includes, p_outlier=0.0, model=model
)
```

### Code cell 8

```python
# Sample
hddmnn_model.sample(nmcmc, burn=100)
```

### Code cell 9

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(12, 8),
    value_range=np.arange(0, 3, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "add_legend": False,
        "alpha": 0.01,
        "ylim": 6.0,
        "bin_size": 0.025,
        "add_posterior_mean_model": True,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "samples": 200,
        "legend_fontsize": 7,
        "legend_loc": "upper left",
        "linewidth_histogram": 1.0,
        "subplots_adjust": {"top": 0.9, "hspace": 0.35, "wspace": 0.3},
    }
)
```

### `caterpillar plot`

The `caterpillar_plot()` function below displays *parameterwise*, 

-  as a <span style="color:blue"> **blue** </span> tick-mark the **ground truth**.
-  as a *thin* **black** line the $1 - 99$ percentile range of the posterior distribution
-  as a *thick* **black** line the $5-95$ percentile range of the posterior distribution

Again use the ```help()``` function to learn more.

### Code cell 12

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    drop_sd=False,
    figsize=(8, 6),
)
```

### `posterior predictive` (standard)

We have two versions of the *standard posterior predictive plots*. Both are called via the `plot_posterior_predictive` function, however you pick the `plot_func` argument accordingly. 

If you pick `hddm.plotting.plot_func_posterior_pdf_node_nn`, the resulting plot uses likelihood *pdf* evaluations to show you the posterior predictive including uncertainty.

If you pick `hddm.plotting.plot_func_posterior_pdf_node_from_sim`, the posterior predictives instead derive from actual simulation runs from the model.

Both function have a number of options to customize styling.

Below we illustrate:

#### `_plot_func_posterior_pdf_node_nn`

### Code cell 16

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(-4, 4, 0.01),
    plot_func=hddm.plotting._plot_func_posterior_pdf_node_nn,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.05,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": True,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
plt.show()
```

#### `_plot_func_posterior_node_from_sim`

### Code cell 18

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(-6, 6, 0.02),
    plot_func=hddm.plotting._plot_func_posterior_node_from_sim,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.1,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": False,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
```

### `posterior predictive` (model cartoon)

The **model plot** is useful to illustrate the behavior of a models pictorially,
including the uncertainty over model parameters embedded in the posterior distribution.

This plot works only for **2-choice** models at this point.

Check out more of it's capabilities with the `help()` function.

### Code cell 21

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(0, 4, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_mean_model": True,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": True,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
plt.show()
```

### `posterior pair plot`

### Code cell 23

```python
hddm.plotting.plot_posterior_pair(
    hddmnn_model, save=False, parameter_recovery_mode=True, samples=200, figsize=(6, 6)
)
```

### NEW (v0.9.3): Plots for $n>2$ choice models

### Code cell 25

```python
# Metadata
nmcmc = 1000
model = "race_no_bias_angle_4"
n_samples = 1000
includes = deepcopy(hddm.simulators.model_config[model]["hddm_include"])
includes.remove("z")
```

### Code cell 26

```python
data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(
    n_subjects=5,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions=None,
    depends_on=None,
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=["z"],
)
```

### Code cell 27

```python
data
```

### Code cell 28

```python
# Define the HDDM model
hddmnn_model = hddm.HDDMnn(
    data, informative=False, include=includes, p_outlier=0.0, model=model
)
```

### Code cell 29

```python
# Sample
hddmnn_model.sample(nmcmc, burn=100)
```

#### `caterpillar plot`

### Code cell 31

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    drop_sd=False,
    figsize=(8, 6),
)
plt.show()
```

#### `posterior_predictive_plot`

### Code cell 33

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(0, 4, 0.02),
    plot_func=hddm.plotting._plot_func_posterior_node_from_sim,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.1,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": False,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
```

#### `model_cartoon_plot`

**WARNING:** The plot below should not be taken as representative for a particular model fit. The chain may need to be run for much longer than the number of samples allotted in this tutorial script.

### Code cell 36

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(12, 8),
    value_range=np.arange(0, 3, 0.1),
    plot_func=hddm.plotting._plot_func_model_n,
    parameter_recovery_mode=True,
    **{
        "add_legend": False,
        "alpha": 0.01,
        "ylim": 6.0,
        "bin_size": 0.025,
        "add_posterior_mean_model": True,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "samples": 200,
        "legend_fontsize": 7,
        "legend_loc": "upper left",
        "linewidth_histogram": 1.0,
        "subplots_adjust": {"top": 0.9, "hspace": 0.35, "wspace": 0.3},
    }
)
plt.show()
```
