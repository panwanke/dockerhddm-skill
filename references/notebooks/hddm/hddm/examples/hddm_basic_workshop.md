# hddm basic workshop

> Converted from `hddm/hddm/examples/hddm_basic_workshop.ipynb`. Code is preserved; rich outputs are omitted.

# HDDM WORKSHOP 
## (Alexander Fengler, May 2022)

## COLAB Installation Instructions (optional)

### Code cell 3

```python
# !pip install cython
# !pip install pymc # if there are problems --> usually here
# !pip install git+https://github.com/hddm-devs/kabuki
# !pip install git+https://github.com/hddm-devs/hddm
# !pip install torch torchvision torchaudio # optional
```

## Basic Preparation

### Code cell 5

```python
# MODULE IMPORTS ----

from copy import deepcopy

# warning settings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Data management
import pandas as pd
import numpy as np
import pickle

# Plotting
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Stats functionality
from statsmodels.distributions.empirical_distribution import ECDF

# HDDM
import hddm
from hddm.simulators.hddm_dataset_generators import simulator_h_c
```

### Initial Examples (most useful functions)

### Data Simulators

#### The `simulator_h_c()` function

This function is useful if you want to create complex datasets (e.g. for parameter recovery studies). To check specifics you can use the `help()` function.

##### Simple example

### Code cell 10

```python
# Simulate some data

# 'ddm_hddm_base' specifies usage of the basic ddm simulator
# recent changes to hddm allow many more models to be fit, necessitating
# explicit naming.
model = "ddm_hddm_base"

n_samples = 1000  # number of samples (trials) the simulated data should contain
n_subjects = 1

data, parameter_dict = simulator_h_c(
    n_subjects=n_subjects,
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

### Code cell 11

```python
data
```

### Code cell 12

```python
# Generating parameters
parameter_dict
```

##### Complex example

### Code cell 14

```python
# Simulate some data

# 'ddm_hddm_base' specifies usage of the basic ddm simulator
# recent changes to hddm allow many more models to be fit, necessitating
# explicit naming.
model = "ddm_hddm_base"

n_samples = 1000  # number of samples (trials) the simulated data should contain
n_subjects = 10

data, parameter_dict = simulator_h_c(
    n_subjects=n_subjects,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions={"my_condition": ["high", "low"]},
    depends_on={"v": ["my_condition"]},
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 15

```python
data
```

### Code cell 16

```python
# Generating parameters
parameter_dict
```

#### The `simulator()` function

This function is useful for low level access to the simulators and full flexbility.
The `simulator()` function provides a `tuple`, with the following elements as output.

- `[0]` provides the *reaction times*
- `[1]` provides the *choices*
- `[2]` provides a bunch of *metadata*

### Code cell 18

```python
# Get some default parameters

# The 'model_config' dictionary contains essential information about all models
# currently included in HDDM
params = hddm.model_config.model_config["ddm_hddm_base"]["params_default"]
params[3] = 0.5
sim_out = hddm.simulators.simulator(theta=params, model=model, n_samples=n_samples)

theta = np.zeros((n_samples, 5))
theta[:, 0] = sim_out[2]["v"][0]
theta[:, 1] = sim_out[2]["a"][0]
theta[:, 2] = sim_out[2]["z"][0]
theta[:, 3] = sim_out[2]["t"][0]
theta[:, 4] = 0

data = pd.DataFrame(
    np.concatenate([(sim_out[0]), (sim_out[1]), theta], axis=1),
    columns=["rt", "response", "v", "a", "z", "t", "subj_idx"],
)
data["subj_idx"] = data["subj_idx"].apply(str)
```

### Code cell 19

```python
sim_out[2]
```

### Code cell 20

```python
# Same format as data above
data
```

##### NOTE:

In the `hddm.generate` and `hddm.simulators.hddm_dataset_generators` modules you will find yet more options for *simulator* functions.

### Plot from Data

#### The `plot_from_data()` function

This function can be used to illustrate various aspects of a dataset.
The example below shows a version that includes a *cartoon of the generative model*,
(this can be very useful e.g. in investigating model behavior across parameter settings)

##### SIMPLE

### Code cell 25

```python
# Plot the dataset
hddm.plotting.plot_from_data(
    df=data,
    generative_model="ddm_hddm_base",
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 3),
    value_range=np.arange(0, 6, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    **{
        "alpha": 1.0,
        "ylim": 3,
        "hist_bottom": 0.0,
        "add_data_rts": True,
        "add_data_model": True,
        "add_data_model_markertype_starting_point": ">",
    }
)
plt.show()
```

We can also just show the *reaction times* and drop the *model cartoon*.

### Code cell 27

```python
# Plot the dataset
hddm.plotting.plot_from_data(
    df=data,
    # generative_model = 'ddm_hddm_base',
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 3),
    value_range=np.arange(0, 6, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    **{
        "alpha": 1.0,
        "ylim": 3,
        "add_data_rts": True,
        "hist_bottom": 0.0,
        "add_data_model": False,  # NOTE WE DO NOT SHOW THE MODEL NOW
    }
)
plt.show()
```

##### COMPLEX

### Code cell 29

```python
# Simulate some data

# 'ddm_hddm_base' specifies usage of the basic ddm simulator
# recent changes to hddm allow many more models to be fit, necessitating
# explicit naming.
model = "ddm_hddm_base"

n_samples = 1000  # number of samples (trials) the simulated data should contain
n_subjects = 10

data, parameter_dict = simulator_h_c(
    n_subjects=n_subjects,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions={"condition": ["high", "low"]},
    depends_on={"v": ["condition"]},
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 30

```python
data
```

### Code cell 31

```python
# Plot the dataset
hddm.plotting.plot_from_data(
    df=data,
    generative_model="ddm_hddm_base",
    columns=4,
    groupby=["condition", "subj_idx"],
    figsize=(13, 20),
    value_range=np.arange(0, 6, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    **{
        "alpha": 1.0,
        "ylim": 3,
        "hist_bottom": 0,  # bottom of upper histogram
        "add_data_rts": True,
        "add_data_model": True,
    }
)
plt.show()
```

##### NOTE

Via the `**kwargs` you have many styling options with this plot.

### A First Model Fit

Let's generate some simple data again.

### Code cell 34

```python
# Get some default parameters

# The 'model_config' dictionary contains essential information about all models
# currently included in HDDM
params = hddm.model_config.model_config["ddm_hddm_base"]["params_default"]
params[3] = 0.5
sim_out = hddm.simulators.simulator(theta=params, model=model, n_samples=n_samples)

theta = np.zeros((n_samples, 5))
theta[:, 0] = sim_out[2]["v"][0]
theta[:, 1] = sim_out[2]["a"][0]
theta[:, 2] = sim_out[2]["z"][0]
theta[:, 3] = sim_out[2]["t"][0]
theta[:, 4] = 0

data = pd.DataFrame(
    np.concatenate([(sim_out[0]), (sim_out[1]), theta], axis=1),
    columns=["rt", "response", "v", "a", "z", "t", "subj_idx"],
)
data["subj_idx"] = data["subj_idx"].apply(str)
```

### Code cell 35

```python
# Define the HDDM model
includes = hddm.model_config.model_config[model]["hddm_include"]
n_models = 2
hddm_models = []

for i in range(n_models):
    hddm_models.append(
        hddm.HDDM(
            data,
            informative=True,
            is_group_model=False,
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )
```

### Code cell 36

```python
# Sample
nmcmc = 2000

for i in range(n_models):
    print(i)
    hddm_models[i].sample(nmcmc, burn=1000)
```

##### NOTE

You can parallelize the model fits using the `multiprocessing` library for example.
See the template below:

```
# imports ...
from functools import partial
from multiprocessing import Pool
import psutil
# etc. etc.

def run_model(chain_id, data, x, y, z):
    import hddm

    # Process x,y,z ...
  
    # Define Model
    model = hddm.HDDM(data,
              informative = x,
              is_group_model = True,
              include = y,
              p_outlier = z)
    
    # Sample
    model.sample(1000, burn = 500, dbname = 'db_name_' + str(chain_id) + '.db', db = 'pickle')
    
    # Save
    model.save('my_model_' + str(chain_id) + '.pickle')
    
    return 0
    
if __name__ == "__main__":
    n_cpus = psutil.cpu_count(logical = False)
    n_chains = ...
    data = ...
    x = ...
    y = ...
    z = ...
    
    run_model_prepped = partial(run_model, data = data, x = x, y = y, z = z)
    
    with Pool(processes = n_cpus) as pool:
        pool_out = pool.map(run_model_prepped, [1, 2, ... , n_chains])
    
    print("Finished")
```

### Reporting HDDM results

#### Posterior Statistics

### Code cell 40

```python
# Posterior Means
hddm_models[0].gen_stats()
```

#### Posterior Samples

### Code cell 42

```python
# A look at the traces
hddm_models[0].get_traces()
```

#### Convergence

### Code cell 44

```python
from kabuki.analyze import gelman_rubin

gelman_rubin(hddm_models)
```

### Code cell 45

```python
# Plot the traces
hddm_models[0].plot_posteriors()
plt.show()
```

#### Other Plots

### Code cell 47

```python
# Posterior Predictive Plots
hddm.plotting.plot_posterior_pair(
    hddm_models[0], parameter_recovery_mode=False, samples=500, figsize=(6, 6)
)
```

### Code cell 48

```python
# Caterpillar plots
hddm.plotting.plot_caterpillar(
    hddm_model=hddm_models[0],
    ground_truth_parameter_dict=None,
    figsize=(5, 5),
    columns=3,
)
```

#### Posterior Predictives

##### Access via plotting: `plot_posterior_predictive()`

### Code cell 51

```python
# Posterior Predictive
# FIX: IMPROVE LABELING
hddm.plotting.plot_posterior_predictive(
    model=hddm_models[0],
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(-15, 15, 0.1),
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.4,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": True,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
plt.show()
```

##### Lower level access: `post_pred_gen()`

### Code cell 53

```python
# Direct access to posterior predictives
posterior_predictive_sample = hddm.utils.post_pred_gen(
    model=hddm_models[0], samples=200, groupby=["subj_idx"], append_data=True
)
```

### Code cell 54

```python
posterior_predictive_sample
```

From here we can compute any quantity we want, comparing any aspect of our original dataset with the synthetic data we can generate via the posterior predictives.

##### Lowest level access: `simulator()`

Using the simulator function gives you the greates amount of flexibility. Just take posterior samples from the traces and use them as parameter (`theta`) inputs to the simulator function directly. There is essentially nothing you shouldn't be able to do.

### More Complex Models

#### Condition / Regression, simplest case

We show how to replicate an analysis using the basic `HDDM()` class, with the `HDDMRegressor()` class.

### Code cell 59

```python
# Simulate some data
model = "ddm_hddm_base"
n_samples = 1000

data, parameter_dict = simulator_h_c(
    n_subjects=1,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions={"my_condition": ["high", "low"]},
    depends_on={"v": ["my_condition"]},
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)

data_new = deepcopy(data)
data_new["my_condition_reg"] = 0
data_new.loc[data_new["my_condition"] == "high", "my_condition_reg"] = 1

data_new
```

##### Using `HDDM()` and `depends_on`

### Code cell 61

```python
hddm_models_condition = []

for i in range(n_models):
    hddm_models_condition.append(
        hddm.HDDM(
            data_new,
            informative=True,
            is_group_model=False,
            depends_on={"v": "my_condition"},
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(1):
    print(i)
    hddm_models_condition[i].sample(nmcmc, burn=1000)
```

### Code cell 62

```python
hddm_models_condition[0].gen_stats()
```

##### Using `HDDMRegressor()`

### Code cell 64

```python
hddm_models_reg = []


def identity_link(x):
    return x


reg_models = [{"model": "v ~ 1 + my_condition_reg", "link_func": identity_link}]

for i in range(n_models):
    hddm_models_reg.append(
        hddm.HDDMRegressor(
            data_new,
            reg_models,
            informative=True,
            is_group_model=False,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(n_models):
    print(i)
    hddm_models_reg[i].sample(nmcmc, burn=1000)
```

### Code cell 65

```python
hddm_models_reg[0].gen_stats()
```

#### Condition / Regression, hierarchical

Replicating the analysis above, but this time using a hierarchical model.

### Code cell 67

```python
# Simulate some data
model = "ddm_hddm_base"
n_samples = 200

data, parameter_dict = simulator_h_c(
    n_subjects=10,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions={"my_condition": ["high", "low"]},
    depends_on={"v": ["my_condition"]},
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)

data_new
data_new = deepcopy(data)

data_new["my_condition_reg"] = 0
data_new.loc[data_new["my_condition"] == "high", "my_condition_reg"] = 1
```

##### Using `HDDM()` and `depends_on`

### Code cell 69

```python
# Define the HDDM model
hddm_models_group_condition = []

for i in range(n_models):
    hddm_models_group_condition.append(
        hddm.HDDM(
            data_new,
            informative=True,
            is_group_model=True,
            depends_on={"v": "my_condition"},
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(1):
    print(i)
    hddm_models_group_condition[i].sample(nmcmc, burn=1000)
```

### Code cell 70

```python
hddm_models_group_condition[0].gen_stats()
```

##### Using `HDDMRegressor()`

### Code cell 72

```python
hddm_models_group_reg = []


def identity_link(x):
    return x


reg_models = [{"model": "v ~ 1 + my_condition_reg", "link_func": identity_link}]

for i in range(n_models):
    hddm_models_group_reg.append(
        hddm.HDDMRegressor(
            data_new,
            reg_models,
            informative=True,
            is_group_model=True,
            group_only_regressors=False,
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(n_models):
    print(i)
    hddm_models_group_reg[i].sample(nmcmc, burn=1000)
```

### Code cell 73

```python
hddm_models_group_reg[0].gen_stats()
```

#### Regression + *depends_on*

### Code cell 75

```python
# Simulate some data
model = "ddm_hddm_base"
n_samples = 200

data, parameter_dict = simulator_h_c(
    n_subjects=10,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions={"my_condition": ["high", "low"]},
    depends_on={"v": ["my_condition"]},
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)

data_new = deepcopy(data)
data_new["my_condition_reg"] = 0
data_new.loc[data["my_condition"] == "high", "my_condition_reg"] = 1
data_new["randcol"] = np.random.uniform(low=-1, high=1, size=data_new.shape[0])
```

### Code cell 76

```python
parameter_dict
```

### Code cell 77

```python
hddm_models_group_reg_depends = []


def identity_link(x):
    return x


reg_models = [{"model": "v ~ 1 + randcol", "link_func": identity_link}]

for i in range(n_models):
    hddm_models_group_reg_depends.append(
        hddm.HDDMRegressor(
            data_new,
            reg_models,
            depends_on={"v": ["my_condition"]},
            informative=True,
            is_group_model=True,
            std_depends=True,
            group_only_regressors=False,
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(n_models):
    print(i)
    hddm_models_group_reg_depends[i].sample(nmcmc, burn=1000)
```

### Code cell 78

```python
hddm_models_group_reg_depends[0].gen_stats()[25:50]
```

### Link Functions

### Code cell 80

```python
# Simulate some data
model = "ddm_hddm_base"
n_samples = 1000

data, parameter_dict = simulator_h_c(
    n_subjects=1,
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

### Code cell 81

```python
parameter_dict
```

##### SOME NOTES:

In principle you are free to choose link function as you please. 

The general model you will fit is of the following form (take the drift parameter as an example):

$$ \hat{v} = link(\mathbf{X} \beta) $$

Using the **identity link** leaves us with,

$$ \hat{v} = \mathbf{X} \beta $$

Using the **logistic link** leave us with,

$$ \hat{v} = \frac{1}{1 + \exp( - \mathbf{X} \beta)} $$

The **identity function** will most often be the right choice.

#### Example 1: Identity Link on z

### Code cell 84

```python
def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


reg_models = [
    {"model": "z ~ 1", "link_func": identity_link},
    {"model": "v ~ 1", "link_func": identity_link},
]

hddm_models_link_id = []
n_models = 1

for i in range(n_models):
    hddm_models_link_id.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 1500
for i in range(1):
    print(i)
    hddm_models_link_id[i].sample(nmcmc, burn=500)
```

### Code cell 85

```python
hddm_models_link_id[0].gen_stats()
```

### Code cell 86

```python
# Posterior Predictive
hddm.plotting.plot_posterior_predictive(
    model=hddm_models_link_id[0],
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(-5, 5, 0.05),
    parameter_recovery_mode=False,
    required_method="random",
    plot_func=hddm.plotting._plot_func_posterior_node_from_sim,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.1,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": True,
        "plot_likelihood_raw": False,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
        "posterior_uncertainty_color": "red",
    }
)
```

#### Example 2: Logit link on z

### Code cell 88

```python
def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


reg_models = [
    {"model": "z ~ 1", "link_func": logit_link},
    {"model": "v ~ 1", "link_func": identity_link},
]

hddm_models_link_logit = []
n_models = 1

for i in range(n_models):
    hddm_models_link_logit.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "z", "t"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 1500
for i in range(1):
    print(i)
    hddm_models_link_logit[i].sample(nmcmc, burn=500)
```

### Code cell 89

```python
hddm_models_link_id[0].gen_stats()
```

##### NOTES:

When the **logit link** is used (any link), we have to transform the **z_Intercept** parameter.

### Code cell 91

```python
z_intercept = hddm_models_link_id[0].gen_stats().loc["z_Intercept", "mean"]

print("Computed z")
print(1 / (1 + np.exp(-z_intercept)))

print("Ground Truth z")
print(parameter_dict["z"])
```

### Code cell 92

```python
# Posterior Predictive
hddm.plotting.plot_posterior_predictive(
    model=hddm_models_link_id[0],
    columns=2,
    figsize=(8, 6),
    value_range=np.arange(-5, 5, 0.05),
    parameter_recovery_mode=False,
    required_method="random",
    plot_func=hddm.plotting._plot_func_posterior_node_from_sim,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "bin_size": 0.1,
        "add_posterior_mean_rts": True,
        "add_posterior_uncertainty_rts": True,
        "plot_likelihood_raw": False,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
        "posterior_uncertainty_color": "red",
    }
)
```

##### NOTES:

If we *do not* use the *identity* link on a parameter, the *prior* on it's *intercept* will default to a *normal distribution*.

If we *do* use the *identity* link on a parameter, the *prior* will be the same as the *informative* prior chosen when estimating this parameter *without a regression model*.

#### Example 3: Less Standard use case - Stimulus Coding

We will refer to the slides to discuss stimulus coding, before proceeding to the example.

### Code cell 96

```python
# Simulate some StimCoding Data
n_samples = 1000
v = np.concatenate([np.ones(500), -np.ones(500)])
a = 3
z = 0.5
t = 1

theta = np.zeros((n_samples, 4))
theta[:, 0] = v
theta[:, 1] = a
theta[:, 2] = z
theta[:, 3] = t
```

### Code cell 97

```python
sim_out = hddm.simulators.simulator(
    theta=theta, model="ddm_hddm_base", n_samples=1, max_t=40
)

data = pd.DataFrame(
    np.hstack([sim_out[0], sim_out[1], theta]),
    columns=["rt", "response", "v", "a", "z", "t"],
)
data["stim"] = 2
data.loc[data["v"] == -1.0, "stim"] = 1
```

##### VIA STIMCODING

### Code cell 99

```python
hddm_models_stim = []
n_models = 1

for i in range(n_models):
    hddm_models_stim.append(
        hddm.HDDMStimCoding(
            data,
            split_param="v",
            stim_col="stim",
            informative=True,
            is_group_model=False,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 1500
for i in range(1):
    print(i)
    hddm_models_stim[i].sample(nmcmc, burn=500)
```

### Code cell 100

```python
hddm_models_stim[0].gen_stats()
```

##### VIA REGRESSION

### Code cell 102

```python
from patsy import dmatrix


def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


def stim_coding(x, data=data):
    stim = np.squeeze(
        (np.asarray(dmatrix("0 + C(s,[[-1],[1]])", {"s": data.stim.loc[x.index]})))
    )
    return x * stim


reg_models = [
    {"model": "z ~ 1", "link_func": identity_link},
    {"model": "v ~ 1", "link_func": stim_coding},
]

hddm_models_stim_reg = []
n_models = 1

for i in range(n_models):
    hddm_models_stim_reg.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(1):
    print(i)
    hddm_models_stim_reg[i].sample(nmcmc, burn=1000)
```

### Code cell 103

```python
hddm_models_stim_reg[0].gen_stats()
```

##### VIA REGRESSION - ALTERNATIVE VERSION (VIA COLUMN TRANSFORM)

### Code cell 105

```python
data["stimalt"] = 1
data.loc[data["stim"] == 2, "stimalt"] = -1
```

### Code cell 106

```python
data
```

### Code cell 107

```python
from patsy import dmatrix


def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


def stim_coding(x, data=data):
    stim = np.squeeze(
        (np.asarray(dmatrix("0 + C(s,[[-1],[1]])", {"s": data.stim.loc[x.index]})))
    )
    return x * stim


def stim_coding_alt(x, data=data):
    return x * data.stimalt.loc[x.index].values


reg_models = [
    {"model": "z ~ 1", "link_func": identity_link},
    {"model": "v ~ 0 + stimalt", "link_func": stim_coding_alt},
]

hddm_models_stim_reg_alt = []
n_models = 1

for i in range(n_models):
    hddm_models_stim_reg_alt.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 1000
for i in range(1):
    print(i)
    hddm_models_stim_reg_alt[i].sample(nmcmc, burn=500)
```

### Code cell 108

```python
hddm_models_stim_reg_alt[0].gen_stats()
```

### Interactions (Continuous Covariates)

### Code cell 110

```python
# Make data
n_trials = 1000

beta_v_intercept = 1
beta_v_cov_1 = 1
beta_v_cov_2 = -1
beta_v_interaction = 0.5
cov_1 = np.random.uniform(low=-1, high=1, size=n_trials)
cov_2 = np.random.uniform(low=-1, high=1, size=n_trials)

v = (
    beta_v_intercept
    + beta_v_cov_1 * cov_1
    + beta_v_cov_2 * cov_2
    + beta_v_interaction * cov_1 * cov_2
)
a = 3
t = 1
z = 0.4

theta = np.zeros((n_trials, 4))
theta[:, 0] = v
theta[:, 1] = a
theta[:, 2] = z
theta[:, 3] = t
```

### Code cell 111

```python
sim_out = hddm.simulators.simulator(
    theta=theta, model="ddm_hddm_base", n_samples=1, max_t=40
)

data = pd.DataFrame(
    np.hstack([sim_out[0], sim_out[1], theta]),
    columns=["rt", "response", "v", "a", "z", "t"],
)
data["cov_1"] = cov_1
data["cov_2"] = cov_2
```

### Code cell 112

```python
data
```

### Code cell 113

```python
# Run Model
def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


reg_models = [
    {"model": "z ~ 1", "link_func": identity_link},
    {"model": "v ~ 1 + cov_1 * cov_2", "link_func": identity_link},
]

hddm_models_reg_interact = []
n_models = 1

for i in range(n_models):
    hddm_models_reg_interact.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(1):
    print(i)
    hddm_models_reg_interact[i].sample(nmcmc, burn=1000)
```

### Code cell 114

```python
hddm_models_reg_interact[0].gen_stats()
```

##### NOTES:

A note on the interpretation of the **beta coefficients**. 
Let's say we as whether our interaction term is "*significantly*" different from 0.

We can plot the trace and signify e.g. it's $5th$ percentile. 
If the $5th$ percentile is above 0 (or the $95th$ percentile below 0) we can conclude with some confidence that 
there is a **positive interaction** (negative interaction) between *cov_1* and *cov_2*.

### Code cell 116

```python
# Bayesian 'hypothesis test' checking whether the interaction is 'significant'

interaction_trace = hddm_models_reg_interact[0].get_traces()["v_cov_1:cov_2"].values
interaction_trace_sorted = np.sort(interaction_trace)

percentile = 0.05
plt.hist(interaction_trace, bins=30, histtype="step", density=True)
plt.axvline(
    x=interaction_trace_sorted[int(np.floor(percentile * interaction_trace.shape[0]))],
    color="red",
    linestyle="dashed",
)
plt.title("Interaction beta posteiror")
plt.xlim((0, 1))
```

### Interactions (Categorical Covariates)

This example comes closest to **ANOVA** style analysis.

### Code cell 119

```python
# Make data
n_trials = 1000

beta_v_intercept = 1
beta_v_cov_1 = 1
beta_v_cov_2 = -1
beta_v_interaction = 0.5
cov_1 = np.concatenate(
    [np.ones(500), np.zeros(500)]
)  # np.random.uniform(low = -1, high = 1, size = n_trials)
cov_2 = np.concatenate(
    [np.ones(250), np.zeros(250), np.ones(250), np.zeros(250)]
)  # np.random.uniform(low = -1, high = 1, size = n_trials)
v = (
    beta_v_intercept
    + beta_v_cov_1 * cov_1
    + beta_v_cov_2 * cov_2
    + beta_v_interaction * cov_1 * cov_2
)
a = 3
t = 1
z = 0.4

theta = np.zeros((n_trials, 4))
theta[:, 0] = v
theta[:, 1] = a
theta[:, 2] = z
theta[:, 3] = t
```

### Code cell 120

```python
sim_out = hddm.simulators.simulator(
    theta=theta, model="ddm_hddm_base", n_samples=1, max_t=40
)

data = pd.DataFrame(
    np.hstack([sim_out[0], sim_out[1], theta]),
    columns=["rt", "response", "v", "a", "z", "t"],
)
data["cov_1"] = cov_1
data["cov_2"] = cov_2
data["cov_1"] = data["cov_1"].apply(str)
data["cov_2"] = data["cov_2"].apply(str)
```

### Code cell 121

```python
data
```

### Code cell 122

```python
# Run Model
def identity_link(x):
    return x


def logit_link(x):
    return 1 / (1 + np.exp(-x))


reg_models = [
    {"model": "z ~ 1", "link_func": identity_link},
    {"model": "v ~ 1 + cov_1 * cov_2", "link_func": identity_link},
]

hddm_models_reg_interact = []
n_models = 1

for i in range(n_models):
    hddm_models_reg_interact.append(
        hddm.HDDMRegressor(
            data,
            reg_models,
            informative=True,
            is_group_model=False,
            group_only_regressors=True,
            include=["v", "a", "t", "z"],
            p_outlier=0,
        )
    )

# Sample
nmcmc = 2000
for i in range(1):
    print(i)
    hddm_models_reg_interact[i].sample(nmcmc, burn=1000)
```

### Code cell 123

```python
hddm_models_reg_interact[0].gen_stats()
```

### Priors

We try to illustrate the effect of choosing between *informative* and *uninformative* priors in your data analysis (and relatedly the added benefit you might expect from introducing more user side freedom on prior choice). We pick a number of different dataset-sizes (trial numbers), run inference *once with informative* and *once with uninformative* priors and then compare the posteriors for each of the parameters **\[v, a, z, t\]**.

**Spoiler-alert**:

Don't expect much !

### Code cell 125

```python
# Generate some data
n_samples = 10000
parameters = hddm.model_config.model_config["ddm_hddm_base"]["params_default"]
parameters[hddm.model_config.model_config["ddm_hddm_base"]["params"].index("t")] = 0.5

theta = np.zeros((n_samples, 4))
theta[:, 0] = parameters[
    hddm.model_config.model_config["ddm_hddm_base"]["params"].index("v")
]
theta[:, 1] = parameters[
    hddm.model_config.model_config["ddm_hddm_base"]["params"].index("a")
]
theta[:, 2] = parameters[
    hddm.model_config.model_config["ddm_hddm_base"]["params"].index("z")
]
theta[:, 3] = parameters[
    hddm.model_config.model_config["ddm_hddm_base"]["params"].index("t")
]

sim_out = hddm.simulators.simulator(
    theta=parameters, model="ddm_hddm_base", n_samples=n_samples, max_t=40
)

data = pd.DataFrame(
    np.hstack([sim_out[0], sim_out[1], theta]),
    columns=["rt", "response", "v", "a", "z", "t"],
)
```

### Code cell 126

```python
# Choose sample sizes for our model fits
sample_sizes = [32, 64, 128, 256, 512, 1024]
stats_list = []
trace_list = []

for n_samples_tmp in sample_sizes:
    print("Sample size: ", n_samples_tmp)
    data_tmp = data.sample(n_samples_tmp).reset_index(drop=True)

    # Informative -----
    hddm_model_tmp = hddm.HDDM(
        data_tmp,
        informative=True,
        is_group_model=False,
        include=["v", "a", "t", "z"],
        p_outlier=0,
    )
    hddm_model_tmp.sample(1000, burn=500)

    # Save traces
    tmp_traces = hddm_model_tmp.get_traces()
    tmp_traces["informative"] = 1
    tmp_traces["sample_size"] = n_samples_tmp
    trace_list.append(deepcopy(tmp_traces))
    # -----

    # Uninformative -----
    hddm_model_tmp = hddm.HDDM(
        data_tmp,
        informative=False,
        is_group_model=False,
        include=["v", "a", "t", "z"],
        p_outlier=0,
    )
    hddm_model_tmp.sample(1000, burn=500)

    # Save traces
    tmp_traces = hddm_model_tmp.get_traces()
    tmp_traces["informative"] = 0
    tmp_traces["sample_size"] = n_samples_tmp
    trace_list.append(deepcopy(tmp_traces))
    # -----

trace_df = pd.concat(trace_list)
trace_df["sample_size"] = trace_df["sample_size"].apply(str)
```

### Code cell 127

```python
for param in ["v", "a", "z_trans", "t"]:
    g = sns.catplot(
        x=param,
        y="sample_size",
        hue="informative",
        kind="violin",
        data=trace_df,
        title=param,
    )
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(param)
```

### END
