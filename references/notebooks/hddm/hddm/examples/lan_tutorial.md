# lan tutorial

> Converted from `hddm/hddm/examples/lan_tutorial.ipynb`. Code is preserved; rich outputs are omitted.

# LAN Tutorial


This tutorial is a rather comprehensive introduction to **HDDM** with focus on the new **LAN extension**.
The methods behind the **new** `HDDMnn()`, `HDDMnnRegressor()` and `HDDMnnStimcoding()` classes can be found in our original dedicated [publication](https://elifesciences.org/articles/65074).
These are new featues. Please let us know on the HDDM forum and/or via github reports regarding bugs or other limitations and we will do our best to help as soon as we can.

## Things to look out for:
 
 - Networks were trained over a fairly wide range of parameters which hopefully capture the scope of common empirical data. The networks will not accurately report likelihoods outside that range, so we explicitly limit the range of parameters that can be sampled from. If you find that your posterior samples reach and get stuck at the allowed parameter bounds (which you will see in the posterior plots), please notify us and we will do our best to provide improved networks over time. 

- You may encounter more print output than with standard HDDM. These are sanity checks and the verbosity will vanish progressively.

## Section 0: Colab Prep (Optional)

### Reminder
In the *upper left* menu click on **Runtime**, then **Change runtime type** and select **GPU** as **hardware accelerator**

### INSTALLATION COLAB: INSTALL SUPPORT LIBRARIES

### Code cell 5

```python
# Note: Usually colab has all other packages which we may use already installed
# The basic configuration of colabs does change over time, so you may have to add
# some install commands here if imports below don't work for package xyz
!pip install scikit-learn
!pip install cython
!pip install pymc==2.3.8
```

### INSTALLATION COLAB:  INSTALL HDDM

### Code cell 7

```python
!pip install -U --no-deps git+https://github.com/hddm-devs/hddm
!pip install -U --no-deps git+https://github.com/hddm-devs/kabuki
```

### Imports

### Code cell 9

```python
# MODULE IMPORTS ----

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

## Section 1: Model Info / Simulation / Basic Plotting

The main concern of this notebook is to present the extended capabilities of the HDDM toolbox as a result of the new `HDDMnn` classes.

Primarily we are interested in the additional models we can now be fit to data. So let's take stock of the models that were added to standard **HDDM**.

### 2-Choice Models

#### ANGLE

A model with a linearly collapsing angle. Adds a parameter $\theta$, which specifies the angle of the bound.

#### WEIBULL

A model that includes a collapsing bound parameterized as the scaled *cdf* of a Weibull distribution.
This adds two parameters to the standard **DDM**, $\alpha$ and $\beta$.

#### LEVY

The Levy model is essentially a standard **DDM** where noise is not driven by a Gaussian distribution, but the noise process is now parameterized by the new parameter $\alpha$, which interpolates between a Gausian $\alpha = 2$ and a Cauchy (heavy tailed) $\alpha = 1$. 

#### ORNSTEIN

This model implements the 2-choice **LCA**, which includes a an inhibition / excitation parameter $g$. 

Find more details on these models in our companion [paper](https://elifesciences.org/articles/65074). 

### 3 / 4-Choice Models

#### NOTE
The addition of *3 choice* and *4 choice* models, comes with slightly more limited functionality as compared to *2 choice* models. Specifically, not all plot-concepts currently standard in **HDDM** translate immediately to models with more choice options. We are trying to align this functionality going forward.

#### LCA (Leaky Competing Accumulator)
Please find the original description in this [paper](https://pubmed.ncbi.nlm.nih.gov/11488378/).

#### RACE
Race models simply take out the mutual and self-inhibition of **LCAs**.

#### ANGLE versions of LCA / RACE
Implements an linearly collapsing bound as above under the respective *2 choice models*

### 1.1 Access Meta-Data

Let's first take a look at some of the useful metadata we can use to set up our models and simulators. 
If we type ```hddm.simulators.model_config```, we get back a dictionary that stores a bunch of information
for each of the models that are currently implemented in HDDM. It lists,

- A ```doc``` string that gives some information about the status of the model as it pertains to it's usability as well as some potential usage tips. Please read the ```doc``` string before using any of the new models.
- The parameter names under ```params```,
- The parameter bounds that where used for training the network under ```param_bounds```
- The boundary_function (```boundary```) 
- Default parameter values (```params_default```). 
- Slice sampler settings by parameter (```slice_widths```)
- Under ```params_trans``` you can choose parameters which will be logit transformed for sampling (order as in ```params```)
- ```choices``` determines valid choice options under the model
- Under  ```hddm_include```, it lists the parameters which we want to include when initializing our HDDM Model with one of the sequential sampling models available.


You won't need most of these options if you are getting started, but they do provide you with useful information and a couple extra degrees of freedom when it comes to optimizing your sampler.

### Code cell 12

```python
# List the models currently available
hddm.model_config.model_config.keys()
```

#### NOTE

You find **two kinds of extra** models which were not mentioned in the model listing above:

1. Experimental models, which eventually will be fully documented (or dropped)
2. `hddm_base` models are used predominantly with the basic `HDDM()` classes. These models are **not** to be used with the `HDDMnn()` classes.

Now taking a closer look at the ```angle``` model

### Code cell 15

```python
# Metadata
model = "ddm"
n_samples = 1000
```

### Code cell 16

```python
# Config for our current model
hddm.model_config.model_config[model]
```

### Code cell 17

```python
# Looking at the doc string before using the model
print(hddm.model_config.model_config[model]["doc"])
```

### 1.2 Generate Data
Let's start by generating some data from the ```angle``` model. For this you have available the ```simulators``` module, specifically we will start with the ```simulator_h_c``` function.
If you are curious about all the capabilities of this function, please check the `help()` function for it.

### Code cell 19

```python
from hddm.simulators.hddm_dataset_generators import simulator_h_c

data, full_parameter_dict = simulator_h_c(
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

A quick look into what the simulator spits out (you can also read about it in the docs).  <br/><br/>
We get back a ```tuple``` of two:

- *First*, a DataFrame which holds a ``rt``, a `response` and a ```subj_idx``` column as well as trial-by-trial ground truth parameters. 

- *Second* a parameter dictionary which has parameter names in accordance with `HDDM()` trace names. This is useful for some of our plots.

### Code cell 21

```python
data
```

### Code cell 22

```python
# Here unspectacularly, parameter names are unchanged
# (single subject fits do not need any parameter name augmentation)
full_parameter_dict
```

### 1.2 First Plot 

Now that we have our simulated data, we look to visualise it.
Let's look at a couple of plots that we can use for this purpose. 

The `HDDM.plotting` module includes the `plot_from_data` function, which allows you to plot 
subsets from a dataset, according to a grouping specified by the `groupby` argument.

The plot creates a `matplotlib.axes` object for each subset, and you can provide a function to manipulate 
this axes object. Some of these *axes manipulators* are provided your you. Here we focus on the 
`_plot_func_model` *axes manipulator* supplied under the `plot_func` argument.

Check out the arguments of `plot_from_data` and `_plot_func_model` using the `help()` function.
You have quite some freedom in styling these plots.

We will refer to this plot as the `model cartoon plot`.

- The top histogram refers to the probability of choosing option $1$ across time.
- The bottom (upside-down) histogram refers to the probability of choosing option $-1$ (may be coded as $0$ as well) across time.

### Code cell 24

```python
hddm.plotting.plot_from_data(
    df=data,
    generative_model=model,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 3),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    **{"alpha": 1.0, "ylim": 3, "add_data_rts": True, "add_data_model": False}
)
plt.show()
```

If we set `add_model = True`, this will add a cartoon of the model on top of the histograms. 

#### CAUTION
This `model cartoon plot` will only work for *2-choice models* for now.

Moreover, often useful for illustration purposes, we can include a bunch of simulations trajectories into the model plot (note the corresponding arguments). Common to all models currently included is their conceptual reliance on there particle trajectories. Reaction times and choices are simulated as *boundary crossings* of these particles. If you don't want to include these trajectories, just set `show_trajectories = False`.

### Code cell 27

```python
hddm.plotting.plot_from_data(
    df=data,
    generative_model=model,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 3),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    **{"alpha": 1.0, "ylim": 3, "add_data_rts": True, "add_data_model": True}
)
plt.show()
```

If you are interested, you can use this plot to investigate the behavior of models across different parameters setups.

## Section 2: Single Subject (or collapsed) Data


Now, we try to fit these models to data! Let's start with an simple dataset. In other words, we have one single participant who provides  $n$ datatpoints (reaction times and choices) from some *two alternative forced choice* task paradigm.


### Note

In this demo we fit to simulated data. This serves as a template, and you can easily adapt it to your needs.

### Code cell 30

```python
# Metadata
nmcmc = 1500
model = "angle"
n_samples = 1000
includes = hddm.model_config.model_config[model]["hddm_include"]
```

### Note

When defining `includes`,
you can also pick only as subset of the parameters suggested under `hddm.model_config.model_config`.

### Code cell 32

```python
# Generate some simulatred data
from hddm.simulators.hddm_dataset_generators import simulator_h_c

data, full_parameter_dict = simulator_h_c(
    n_subjects=1,
    n_trials_per_subject=n_samples,
    model=model,
    p_outlier=0.00,
    conditions=None,
    depends_on=None,
    regression_models=None,
    regression_covariates=None,  # need this to make initial covariate matrix from which to use dmatrix (patsy)
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 33

```python
data
```

### Code cell 34

```python
# Define the HDDM model
hddmnn_model = hddm.HDDMnn(
    data,
    informative=False,
    include=includes,
    p_outlier=0.01,
    w_outlier=0.1,
    model=model,
)
```

### Code cell 35

```python
# Sample
hddmnn_model.sample(nmcmc, burn=500)
```

### 2.1 Visualization

The `plot_caterpillar()` function below displays *parameterwise*, 

-  as a <span style="color:blue"> **blue** </span> tick-mark the **ground truth**.
-  as a *thin* **black** line the $1 - 99$ percentile range of the posterior distribution
-  as a *thick* **black** line the $5-95$ percentile range of the posterior distribution

Again use the ```help()``` function to learn more.

### Code cell 38

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    figsize=(8, 5),
    columns=3,
)

plt.show()
```

### 2.1.1 Posterior Predictive (via `model cartoon plot`)
Another way to examine whether or not our recovery was satisfactory is to perform posterior predictive checks. Essentially, we are looking to simulate datasets from the trace and check whether it aligns with the ground truth participant data. This answers the question of whether or not these parameters that you recovered can actually reproduce the data. 

Use the `plot_posterior_predictive()` function in the `plotting` module for this. It is structured just like the `plot_from_data()` function, but instead of providing a *dataset*, you supply a *hddm model*.

Use the `help()` function to check out all the functionality.

### Code cell 40

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=1,
    groupby=["subj_idx"],
    figsize=(6, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{"alpha": 0.01, "ylim": 3, "samples": 200}
)
plt.show()
```

**A small note on convergence**:

Note that the MCMC algorithm requires the chain to converge. There are many heuristics that help you identifying problems with convergence, such as the trace plot, auto correlation plot, and marginal posterior histogram. In the trace plots, there might be a problem if you see large jumps. In the autocorrelation plot, there might be a problem if it does not drop rapidly. The `HDDMnn()` classes support the computation of the *Gelman-Rubin*, *r-hat* statistic, as you would with any `hddm` model. Generally, by extracting the traces, you are free to compute any convergence statistics you want of course.

### Code cell 42

```python
# TAKING A LOOK AT THE POSTERIOR TRACES
hddmnn_model.plot_posteriors(hddm.simulators.model_config[model]["params"])
plt.show()
```

### Code cell 43

```python
hddm.plotting.plot_posterior_pair(
    hddmnn_model, save=False, parameter_recovery_mode=True, samples=500, figsize=(6, 6)
)
```

## Section 3: Hierarchical Models

The 'h' in `hddm` stands for hierarchical, so let's do it! If we have data from multiple participants and we assume that the parameters of single participants are drawn from respective **group** or **global** distributions, we can model this explicitly in `hddm` by specifying `is_group_model = True`.

Implicitly we are fitting a model of the following kind,

$$p(\{\theta_j\}, \{\theta_g\} | \mathbf{x}) \propto \left[ \prod_j^{J} \left[ \prod_i^{N_j} p(x_i^j | \theta_j) \right] p(\theta_j | \theta_g) \right] p( \theta_g | \theta_h )$$

where (let's say for the **angle model**),

1. $\theta_j = \{v_j, a_j, z_j, t_j, \theta_j \}$, are the model parameters for **subject j**.

2. $\theta_g = \{v_g^{\mu}, a_g^{\mu}, z_g^{\mu}, t_g^{\mu}, \theta_g^{\mu}, v_g^{\sigma}, a_g^{\sigma}, z_g^{\sigma}, t_g^{\sigma}, \theta_g^{\sigma} \}$ (scary, but for completeness), are the **mean** and **variance** parameters for our group level normal distributions, and $\{ \theta_h \}$ are **fixed hyperparameters**.

3. $x_i^j = \{rt_i^j, c_i^j \}$, are the **choice and reaction time**  of **subject j** during **trial i**.

In words, the right hand side of the equation tells us that we have a **global parameter distribution** with certain **means** and **variances** for each parameter (we want to figure these means and variances out), from which the **subject level parameters** are drawn and finally **subject level datapoints** follow the likelihood distribution of our **ddm / angle / weibull / you name it** mdoels.

### Code cell 45

```python
# Metadata
nmcmc = 1000
model = "angle"
n_trials_per_subject = 200
n_subjects = 10
```

### Code cell 46

```python
# test regressors only False
# add p_outliers to the generator !
data, full_parameter_dict = simulator_h_c(
    data=None,
    n_subjects=n_subjects,
    n_trials_per_subject=n_trials_per_subject,
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

### Code cell 47

```python
hddmnn_model = hddm.HDDMnn(
    data,
    model=model,
    informative=False,
    is_group_model=True,
    include=hddm.simulators.model_config[model]["hddm_include"],
    p_outlier=0.0,
)
```

### Code cell 48

```python
hddmnn_model.sample(
    nmcmc, burn=100
)  # if you want to save the model specify extra arguments --> dbname='traces.db', db='pickle'. # hddmnn_model.save('test_model')
```

### Code cell 49

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    figsize=(8, 5),
    columns=3,
)

plt.show()
```

### Code cell 50

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=3,
    figsize=(10, 7),
    groupby=["subj_idx"],
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_mean_rts": True,
        "add_posterior_mean_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_uncertainty_model": False,
        "samples": 200,
        "legend_fontsize": 7.0,
    }
)
```

## Section 4: Parameter varies by Condition

An important aspect of these posterior analysis, is the consideration of experiment design.
We may have an experiment in which subject are exposed to a variety of conditions, 
such as for example different degrees of difficulty of the same task

It is often reasonable to assume that all but the conceptually relevant parameters are common across conditions.

As a by-product, such experiment designs can help us with the recovery of the constant parameters, by probing those static aspects of the model across varying kinds of datasets (driven by targeted manipulation of variable aspects of the model).

Implicitly we fit the following kind of model,

$$p( \{\theta_c \}, \theta | \mathbf{x} ) \propto  \left[ \prod_c^C  \left[ \prod_i^{N_i} p( x_i^c | \theta_c, \theta ) \right] p(\theta_c)  \right] p(\theta)$$


Where $\theta_c$ is the condition dependent part of the parameter space, and $\theta$ forms the portion of parameters which remain constant across condtions. 

To give a more concrete example involving the **weibull model**, consider a dataset for a single participant, who went through four conditions of an experiment. Think of the conditions as manipulating the payoff structure of the experiment to incentivize / disincentivize accuracy in favor of speed. We operationalize this by treating the $a$ parameter, the initial boundary separation, as affected by the manipulation, while the rest of the parameters are constant across all experiment conditions.

The resulting model would be of the form, 

$$ p( {a_c}, v, z, t, \alpha, \beta | x ) \propto \left[ \prod_c^C  \left[ \prod_i^{N_c} p( x_i^c | a_c, v, z, t, \alpha, \beta)  \right] p(a_c) \right]  p(v, z, t, \alpha, \beta)$$

### Code cell 53

```python
# Metadata
nmcmc = 1000
model = "angle"
n_trials_per_subject = 500

# We allow the boundary conditions to vary
depends_on = {"a": ["c_one"]}

# They will depend on a fictious column 'c_one' that specifies
# levels / conditions
conditions = {"c_one": ["low", "medium", "high"]}
```

### Code cell 54

```python
data, full_parameter_dict = simulator_h_c(
    n_subjects=1,
    n_trials_per_subject=n_trials_per_subject,
    model=model,
    p_outlier=0.00,
    conditions=conditions,
    depends_on=depends_on,
    regression_models=None,
    regression_covariates=None,
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 55

```python
# Let's check the resulting parameter vector
full_parameter_dict
```

### Code cell 56

```python
# Make HDDM Model
hddmnn_model = hddm.HDDMnn(
    data,
    model=model,
    informative=False,
    include=hddm.simulators.model_config[model]["hddm_include"],
    p_outlier=0.0,
    is_group_model=False,
    depends_on=depends_on,
)
```

### Code cell 57

```python
# Sample
hddmnn_model.sample(nmcmc, burn=100)
```

### Code cell 58

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    figsize=(8, 5),
    columns=3,
)

plt.show()
```

### Code cell 59

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_rts": True,
        "add_posterior_uncertainty_model": True,
        "samples": 200,
    }
)
plt.show()
```

### 4.1 Combine Hierarchical and Condition data

### Code cell 61

```python
# Metadata
nmcmc = 1500
model = "angle"
n_subjects = 5
n_trials_per_subject = 500
```

### Code cell 62

```python
data, full_parameter_dict = simulator_h_c(
    n_subjects=n_subjects,
    n_trials_per_subject=n_trials_per_subject,
    model=model,
    p_outlier=0.00,
    conditions={
        "c_one": ["low", "medium", "high"]
    },  # , 'c_three': ['low', 'medium', 'high']},
    depends_on={
        "v": ["c_one"]
    },  # 'theta': ['c_two']}, # 'theta': ['c_two']}, #regression_models = None, #
    regression_models=None,  # regression_covariates = None,
    regression_covariates=None,  # need this to make initial covariate matrix from which to use dmatrix (patsy)
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 63

```python
# Make HDDM Model
hddmnn_model = hddm.HDDMnn(
    data,
    model=model,
    informative=False,
    include=hddm.simulators.model_config[model]["hddm_include"],
    p_outlier=0.0,
    is_group_model=True,
    depends_on={"v": "c_one"},
)
```

### Code cell 64

```python
hddmnn_model.sample(nmcmc, burn=100)
```

### Code cell 65

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=full_parameter_dict,
    figsize=(8, 8),
    columns=3,
)

plt.show()
```

### Code cell 66

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model,
    columns=2,  # groupby = ['subj_idx'],
    figsize=(8, 6),
    value_range=np.arange(1, 2.5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_rts": True,
        "add_posterior_uncertainty_model": True,
        "samples": 200,
        "legend_fontsize": 7,
    }
)
plt.show()
```

## Section 5: Regressors

This section provides a simple working example using the Neural Networks with the Regression backend.
The regression back-end allows linking parameters to trial-by-trial covariates via a (general) linear model.

### Code cell 69

```python
# Metadata
nmcmc = 1000
model = "angle"
n_samples_by_subject = 500
```

### Code cell 70

```python
from hddm.simulators.hddm_dataset_generators import simulator_h_c

data, full_parameter_dict = simulator_h_c(
    n_subjects=5,
    n_samples_by_subject=n_samples_by_subject,
    model=model,
    p_outlier=0.00,
    conditions=None,
    depends_on=None,
    regression_models=["t ~ 1 + covariate_name", "v ~ 1 + covariate_name"],
    regression_covariates={"covariate_name": {"type": "continuous", "range": (0, 1)}},
    group_only_regressors=False,
    group_only=None,
    fixed_at_default=None,
)
```

### Code cell 71

```python
# Set up the regressor a regressor:
reg_model_v = {"model": "v ~ 1 + covariate_name", "link_func": lambda x: x}
reg_model_t = {"model": "t ~ 1 + covariate_name", "link_func": lambda x: x}
reg_descr = [reg_model_t, reg_model_v]
```

### Code cell 72

```python
# Make HDDM model
hddmnn_reg = hddm.HDDMnnRegressor(
    data,
    reg_descr,
    include=hddm.simulators.model_config[model]["hddm_include"],
    model=model,
    informative=False,
    p_outlier=0.0,
)
```

### Code cell 73

```python
# Sample
hddmnn_reg.sample(nmcmc, burn=100)
```

### Code cell 74

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_reg,
    ground_truth_parameter_dict=full_parameter_dict,
    figsize=(8, 8),
    columns=3,
)

plt.show()
```

## Section 6: Stim Coding

You can read more about **stimulus coding** in the [documentation](https://hddm.readthedocs.io/en/latest/howto.html?highlight=stimulus%20coding#code-subject-responses).

Here just an example.

### Code cell 77

```python
# Metadata
nmcmc = 300
model = "ddm"
n_samples_by_condition = 500
split_param = "v"
```

### Code cell 78

```python
sim_data_stimcoding, parameter_dict = hddm.simulators.simulator_stimcoding(
    model=model, split_by=split_param, drift_criterion=0.3, n_trials_per_condition=500
)
```

### Code cell 79

```python
sim_data_stimcoding
```

### Code cell 80

```python
parameter_dict
```

### Code cell 81

```python
hddmnn_model = hddm.HDDMnnStimCoding(
    sim_data_stimcoding,
    include=hddm.simulators.model_config[model]["hddm_include"],
    model=model,
    stim_col="stim",
    p_outlier=0.0,
    split_param=split_param,
    informative=False,
    drift_criterion=True,
)
```

### Code cell 82

```python
hddmnn_model.sample(nmcmc, burn=100)
```

### Code cell 83

```python
hddmnn_model.gen_stats()
```

### Code cell 84

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(
    hddm_model=hddmnn_model,
    ground_truth_parameter_dict=parameter_dict,
    figsize=(8, 5),
    columns=3,
)

plt.show()
```

**NOTE**: 

The `hddm.plotting.plot_posterior_predictive()` does not yet accept *stimcoding* data. This will be updated as soon as possible.

## Section 7: Model Recovery

A crucial exercise in statistical modeling concern **model comparison**. 

We are going to look at model recovery, in this section: Attempt to recover which model generated a given dataset from a set of *candidate models*. 

For the little model recovery study we conduct here, we generate data from the **weibull** model and fit the data once each to the  **weibull**, **angle** and **ddm** models.

We inspect the fits visually and then use the *DIC* (Deviance information criterion, lower is better :)), to check if we can recover the **true** model.

### Code cell 87

```python
# Metadata
model = "weibull"
n_samples = 300
```

### Code cell 88

```python
# test regressors only False
# add p_outliers to the generator !
data, full_parameter_dict = simulator_h_c(
    n_subjects=1,
    n_samples_by_subject=n_samples,
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

### Code cell 89

```python
data
```

### Code cell 90

```python
# Now we fit for each model:
hddmnn_model_weibull = hddm.HDDMnn(
    data,
    informative=False,
    model="weibull",
    p_outlier=0.0,
    include=hddm.simulators.model_config["weibull_cdf"]["hddm_include"],
    is_group_model=False,
)

hddmnn_model_angle = hddm.HDDMnn(
    data,
    model="angle",
    informative=False,
    p_outlier=0.0,
    include=hddm.simulators.model_config["angle"]["hddm_include"],
    is_group_model=False,
)

hddmnn_model_ddm = hddm.HDDMnn(
    data,
    informative=False,
    model="ddm",
    p_outlier=0.0,
    include=hddm.simulators.model_config["ddm"]["hddm_include"],
    is_group_model=False,
)
```

### Code cell 91

```python
nmcmc = 1000
hddmnn_model_weibull.sample(nmcmc, burn=200)

hddmnn_model_angle.sample(nmcmc, burn=200)

hddmnn_model_ddm.sample(nmcmc, burn=200)
```

### 7.1  Checking Model Fits Visually

Posterior Predictive: Do the 'Posterior Models' also make sense?

### Code cell 93

```python
# WEIBULL
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model_weibull,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=True,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_mean_rts": True,
        "samples": 200,
    }
)
plt.show()
```

### Code cell 94

```python
# ANGLE
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model_angle,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=False,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_mean_rts": True,
        "samples": 200,
    }
)
plt.show()
```

### Code cell 95

```python
# DDM
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model_ddm,
    columns=1,
    groupby=["subj_idx"],
    figsize=(4, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=False,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_mean_rts": True,
        "samples": 200,
    }
)
plt.show()
```

### 7.2 Comparing DIC's

### Code cell 97

```python
hddmnn_model_weibull.dic
```

### Code cell 98

```python
hddmnn_model_angle.dic
```

### Code cell 99

```python
hddmnn_model_ddm.dic
```

**Fingers crossed** (this was a random run after all), the DIC usually gives us a result that conforms with the intuition we get from looking at the model plots.

## Section 8: Real Data!

### Code cell 102

```python
# Metadata
nmcmc = 1000
burn = 500
model = "angle"
```

### 8.1 Load and Pre-process dataset

### Code cell 104

```python
# Load one of the datasets shipping with HDDM
cav_data = hddm.load_csv(hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv")
```

### Code cell 105

```python
cav_data
```

### 8.2 Basic Condition Split Model

### Code cell 107

```python
hddmnn_model_cav = hddm.HDDMnn(
    cav_data,
    model=model,
    informative=False,
    include=hddm.simulators.model_config[model]["hddm_include"],
    p_outlier=0.05,
    is_group_model=False,
    depends_on={"v": "stim"},
)
```

### Code cell 108

```python
hddmnn_model_cav.sample(nmcmc, burn=burn)
```

### Code cell 109

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model_cav,
    columns=1,
    figsize=(4, 4),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=False,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_mean_rts": True,
        "samples": 200,
    }
)
plt.show()
```

### 8.3 Basic Hierarchical Model

### Code cell 111

```python
hddmnn_model_cav = hddm.HDDMnn(
    cav_data,
    model=model,
    informative=False,
    include=hddm.simulators.model_config[model]["hddm_include"],
    is_group_model=True,
    p_outlier=0.05,
)
```

### Code cell 112

```python
hddmnn_model_cav.sample(nmcmc, burn=burn)
```

### Code cell 113

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model=hddmnn_model_cav, figsize=(8, 8), columns=3)

plt.show()
```

### Code cell 114

```python
hddm.plotting.plot_posterior_predictive(
    model=hddmnn_model_cav,
    columns=3,
    figsize=(10, 10),
    value_range=np.arange(0, 5, 0.1),
    plot_func=hddm.plotting._plot_func_model,
    parameter_recovery_mode=False,
    **{
        "alpha": 0.01,
        "ylim": 3,
        "add_posterior_uncertainty_model": True,
        "add_posterior_uncertainty_rts": False,
        "add_posterior_mean_rts": True,
        "samples": 200,
        "legend_fontsize": 7,
        "subplots_adjust": {"top": 0.9, "hspace": 0.3, "wspace": 0.3},
    }
)
plt.show()
```

### Note

This is just an example. The angle model might not be the best choice here, and we are moreover ignoring the supplied conditions.

## Section 9: Accessing the Neural Network Directly

The ```network_inspectors``` module allows you to inspect the LANs directly.

### 9.1 Direct access to batch predictions

You can use the ```hddm.network_inspectors.get_torch_mlp()``` function to access network predictions.

### Code cell 120

```python
model = "angle"
```

### Code cell 121

```python
lan_angle = hddm.network_inspectors.get_torch_mlp(model=model)
```

Let's predict some likelihoods !

### Code cell 123

```python
# Make some random parameter set
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model=model, param_dict=None, n_parameter_vectors=1
)
parameter_matrix = np.tile(np.squeeze(parameter_df.values), (200, 1))

# Initialize network input
network_input = np.zeros(
    (parameter_matrix.shape[0], parameter_matrix.shape[1] + 2)
)  # Note the + 2 on the right --> we append the parameter vectors with reaction times (+1 columns) and choices (+1 columns)

# Add reaction times
network_input[:, -2] = np.linspace(0, 3, parameter_matrix.shape[0])

# Add choices
network_input[:, -1] = np.repeat(np.random.choice([-1, 1]), parameter_matrix.shape[0])

# Convert to float
network_input = network_input.astype(np.float32)
# Show example output
print(lan_angle(network_input)[:10])  # printing the first 10 outputs
print(lan_angle(network_input).shape)  # original shape of output
```

### 9.2 Plotting Utilities

HDDM provides two plotting function to investigate the network outputs directly. The ```kde_vs_lan_likelihoods()``` plot and the ```lan_manifold()``` plot.

#### 9.2.1 ```kde_vs_lan_likelihoods()```

The ```kde_vs_lan_likelihoods()``` plot allows you to check the likelihoods produced by a LAN against Kernel Density Estimates (KDEs) from model simulations.
You can supply a panda ```DataFrame``` that holds parameter vectors as rows.

### Code cell 127

```python
# Make some parameters
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model=model, param_dict=None, n_parameter_vectors=10
)
```

### Code cell 128

```python
parameter_df
```

### Code cell 129

```python
hddm.network_inspectors.kde_vs_lan_likelihoods(
    parameter_df=parameter_df, model=model, cols=3, n_samples=2000, n_reps=10, show=True
)
```

#### 9.2.2 ```lan_manifold()```

Lastly, you can use the ```lan_manifold()``` plot to investigate the LAN likelihoods over a range of parameters. 

The idea is to use a base parameter vector and vary one of the parameters in a prespecificed range. 

This plot can be informative if you would like to understand better how a parameter affects model behavior.

### Code cell 131

```python
# Make some parameters
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model=model, param_dict=None, n_parameter_vectors=1
)
```

### Code cell 132

```python
parameter_df
```

### Code cell 133

```python
# Now plotting
hddm.network_inspectors.lan_manifold(
    parameter_df=parameter_df,
    vary_dict={"v": np.linspace(-2, 2, 20)},
    model=model,
    n_rt_steps=300,
    fig_scale=1.0,
    max_rt=5,
    save=True,
    show=True,
)
```

Hopefully this tutorial proves as a useful starting point for your application.
