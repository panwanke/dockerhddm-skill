# lan new models

> Converted from `hddm/hddm/examples/lan_new_models.ipynb`. Code is preserved; rich outputs are omitted.

## New Models

### Code cell 2

```python
import hddm
```

From version HDDM >= 0.9.0, you have access to multiple **new sequential sampling models**. 
You can simulate from these models, perform parameter estimation and moreover you have some extended plotting capabilities which can be useful to visualize model fits, or simply to investigate the behavior of models across parameter settings.

### Metadata

Lets take a look at the new `hddm.model_config.model_config` dictionary, which allows you to investigate metadata for all the new (and old) models which are available through the HDDM-LAN extension.

### Code cell 6

```python
# List all available models
list(hddm.model_config.model_config.keys())[:10]
```

### Code cell 7

```python
# Take an example to list data available for a given model
model_tmp = "ornstein"
hddm.model_config.model_config[model_tmp]
```

You have access to the following data (we focus on the parts important for the user):

- `params`, the names of paramaters for a given model (order matters)
- `params_trans` whether HDDM should internally transform a parameter to an unconstrained domain
- `param_bounds` the range of parameter values that the respective LAN was trained on (order as in `params`) 
- `boundary` the boundary function, which corresponds to the model (access the available boundary functions through the `hddm.simulators.boundary_functions` module.
- `params_default`, defaults settings for the parameters of the model
- `hddm_include`, list to supply to hddm to include all model parameters (you may want to drop some)
-  `slide_widths`, slice sampler settings parameter by parameter (changing these can improve / deteriorate sampler behavior)

You can change these settings as you see fit.

### Simulate

The new `simulator_h_c()` function lets you generate complex datasets using the models available under `hddm.model_config.model_config`. The function is especially useful for parameter recovery studies. It can generate fully synthetic data, or you can supply an empirial dataset and it's structure can be used to generate simulation based replicas. Find more information using the `help()` function. 
Here we give a simple example.

### Code cell 11

```python
model = "angle"
n_subjects = 1
n_samples_by_subject = 500

data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(
    n_subjects=n_subjects,
    n_samples_by_subject=n_samples_by_subject,
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

### Code cell 12

```python
# A look at the data generated
data
```

**Note**:

The full_parameter_dict returned plays well with HDDM and some plots that give you the option
to provide ground truth parameters. In our case the output is simple. 

More complicated datasets, will make this much more interesting.

### Code cell 14

```python
full_parameter_dict
```
