# LAN Tutorial

> Converted from `OfficialTutorials/LAN_Tutorial.ipynb`. Code is preserved; rich outputs are omitted.

# Tutorial

Here we reproduce the LAN tutorial from official website: https://hddm.readthedocs.io/en/latest/lan_tutorial.html#tutorial


Please check the website for details instructions

Thanks to [zenkavi (Zeynep Enkavi)](https://github.com/zenkavi), her repo **intro_HDDM**, which provides the `LAN_Tutorial.ipynb` that we have incorporated it into our project.

Thanks to [Epool (Wanke Pan)](https://github.com/Asynchro-Epool) for conducting thorough testing on the file.

### Code cell 2

```python
# warning settings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
```

### Code cell 3

```python
import hddm
from hddm.simulators.hddm_dataset_generators import simulator_h_c
import kabuki
print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
```

## Section 1: Model Info / Simulation / Basic Plotting

### 1.1 Access Meta-Data

### Code cell 6

```python
# List the models currently available
hddm.model_config.model_config.keys()
```

#### NOTE

You find two kinds of extra models which were not mentioned in the model listing above:

    Experimental models, which eventually will be fully documented (or dropped)

    vanilla models are used predominantly with the basic HDDM() classes. These models are not to be used with the HDDMnn() classes.

#### Now taking a closer look at the angle model

### Code cell 8

```python
# Metadata
model = 'angle'
n_samples = 1000
```

### Code cell 9

```python
# Config for our current model
hddm.model_config.model_config[model]
```

### Code cell 10

```python
# Looking at the doc string before using the model
print(hddm.model_config.model_config[model]['doc'])
```

### 1.2 Generate Data

### Code cell 12

```python
data, full_parameter_dict = simulator_h_c(n_subjects = 1,
                                          n_trials_per_subject = n_samples,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = None,
                                          depends_on = None,
                                          regression_models = None,
                                          regression_covariates = None,
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = ['v', 'a', 'z', 't', 'theta'])
```

### Code cell 13

```python
data
```

### Code cell 14

```python
full_parameter_dict
```

### 1.2 First Plot

### Code cell 16

```python
# help(hddm.plotting.plot_from_data)
# help(hddm.plotting._plot_func_model)
hddm.plotting.plot_from_data(df = data,
                             generative_model = model,
                             columns = 1,
                             groupby = ['subj_idx'],
                             figsize = (4, 3),
                             value_range = np.arange(0, 5, 0.1),
                             plot_func = hddm.plotting._plot_func_model,
                             hist_bottom = 0, # this makes the histograms start from 0. Otherwise https://github.com/hddm-devs/hddm/blob/master/hddm/plotting.py#L1080 makes them start from 2
                             **{'alpha': 1.,
                                'ylim': 3,
                                'add_data_rts': True,
                                'add_data_model': False})
plt.show()
```

### Code cell 17

```python
hddm.plotting.plot_from_data(df = data,
                             generative_model = model,
                             columns = 1,
                             groupby = ['subj_idx'],
                             figsize = (4, 3),
                             value_range = np.arange(0, 5, 0.1),
                             plot_func = hddm.plotting._plot_func_model,
                             hist_bottom = 0,
                             **{'alpha': 1.,
                                'ylim': 3,
                                'add_data_rts': True,
                                'add_data_model': True})
plt.show()
```

## Section 2: Single Subject (or collapsed) Data

### Code cell 19

```python
# Metadata
nmcmc = 1500
model = 'angle'
n_samples = 1000
includes = hddm.model_config.model_config[model]['hddm_include']
```

### Code cell 20

```python
data, full_parameter_dict = simulator_h_c(n_subjects = 1,
                                          n_trials_per_subject = n_samples,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = None,
                                          depends_on = None,
                                          regression_models = None,
                                          regression_covariates = None, # need this to make initial covariate matrix from which to use dmatrix (patsy)
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = None)
```

### Code cell 21

```python
data
```

### Code cell 22

```python
# Define the HDDM model
hddmnn_model = hddm.HDDMnn(data,
                           informative = False,
                           include = includes,
                           p_outlier = 0.01,
                           w_outlier = 0.1,
                           model = model,)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more time**

### Code cell 24

```python
%%time
# Sample
hddmnn_model.sample(
    nmcmc,burn = 500,
    dbname='hddm_single.db', db='pickle'
) 
hddmnn_model.save('hddm_single')
```

### 2.1 Visualization

### Code cell 26

```python
# Caterpillar Plot: (Parameters recovered ok?)
# help(hddm.plotting.plot_caterpillar)# 
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model,
                               ground_truth_parameter_dict = full_parameter_dict,
                               figsize = (8, 5),
                               columns = 3)

plt.show()
```

#### 2.1.1 Posterior Predictive (via `model cartoon plot`)

### Code cell 28

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model,
                                        columns = 1,
                                        groupby = ['subj_idx'],
                                        figsize = (6, 4),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        hist_bottom = 0,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'samples': 200})
plt.show()
```

**A small note on convergence:**

### Code cell 30

```python
# TAKING A LOOK AT THE POSTERIOR TRACES
hddmnn_model.plot_posteriors(hddm.simulators.model_config[model]['params'])
plt.show()
```

### Code cell 31

```python
hddm.plotting.plot_posterior_pair(hddmnn_model, save = False,
                                  parameter_recovery_mode = True,
                                  samples = 500,
                                  figsize = (6, 6))
```

## Section 3: Hierarchical Models

### Code cell 33

```python
# Metadata
nmcmc = 1000
model = 'angle'
n_trials_per_subject = 200
n_subjects = 10
```

### Code cell 34

```python
# test regressors only False
# add p_outliers to the generator !
data, full_parameter_dict = simulator_h_c(data = None,
                                          n_subjects = n_subjects,
                                          n_trials_per_subject = n_trials_per_subject,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = None,
                                          depends_on = None,
                                          regression_models = None,
                                          regression_covariates = None,
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = None)
```

### Code cell 35

```python
hddmnn_model = hddm.HDDMnn(data,
                           model = model,
                           informative = False,
                           is_group_model = True,
                           include = hddm.simulators.model_config[model]['hddm_include'],
                           p_outlier = 0.0)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more than 30 minutes**

### Code cell 37

```python
%%time
hddmnn_model.sample(
    nmcmc,burn = 100,
    dbname='hddm_hier.db', db='pickle'
) 
hddmnn_model.save('hddm_hier')
# if you want to save the model specify extra arguments --> dbname='traces.db', db='pickle'. # hddmnn_model.save('test_model')
```

### Code cell 38

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model,
                               ground_truth_parameter_dict = full_parameter_dict,
                               figsize = (8, 5),
                               columns = 3)

plt.show()
```

### Code cell 39

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model,
                                        columns = 3,
                                        figsize = (10, 7),
                                        groupby = ['subj_idx'],
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        hist_bottom = 0,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_mean_rts': True,
                                        'add_posterior_mean_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_uncertainty_model': False,
                                        'samples': 200,
                                        'legend_fontsize': 7.})
```

## Section 4: Parameter varies by Condition

### Code cell 41

```python
# Metadata
nmcmc = 1000
model = 'angle'
n_trials_per_subject = 500

# We allow the boundary conditions to vary
depends_on = {'a': ['c_one']}

# They will depend on a fictious column 'c_one' that specifies
# levels / conditions
conditions = {'c_one': ['low', 'medium', 'high']}
```

### Code cell 42

```python
data, full_parameter_dict = simulator_h_c(n_subjects = 1,
                                          n_trials_per_subject = n_trials_per_subject,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = conditions,
                                          depends_on = depends_on,
                                          regression_models = None,
                                          regression_covariates = None,
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = None)
```

### Code cell 43

```python
# Let's check the resulting parameter vector
full_parameter_dict
```

### Code cell 44

```python
# Make HDDM Model
hddmnn_model = hddm.HDDMnn(data,
                           model = model,
                           informative = False,
                           include = hddm.simulators.model_config[model]['hddm_include'],
                           p_outlier = 0.0,
                           is_group_model = False,
                           depends_on = depends_on)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more than 10 minutes**

### Code cell 46

```python
%%time
# Sample
hddmnn_model.sample(
    nmcmc,burn = 100,
    dbname='hddm_by_con.db', db='pickle'
) 
hddmnn_model.save('hddm_by_con')
```

### Code cell 47

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model,
                               ground_truth_parameter_dict = full_parameter_dict,
                               figsize = (8, 5),
                               columns = 3)

plt.show()
```

### Code cell 48

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model,
                                        columns = 1,
                                        groupby = ['subj_idx'],
                                        figsize = (4, 4),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_rts': True,
                                        'add_posterior_uncertainty_model': True,
                                        'samples': 200})
plt.show()
```

### 4.1 Combine hierarchical and condition data

### Code cell 50

```python
# Metadata
nmcmc = 1500
model = 'angle'
n_subjects = 5
n_trials_per_subject = 500
```

### Code cell 51

```python
data, full_parameter_dict = simulator_h_c(n_subjects = n_subjects,
                                          n_trials_per_subject = n_trials_per_subject,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = {'c_one': ['low', 'medium', 'high']}, #, 'c_three': ['low', 'medium', 'high']},
                                          depends_on = {'v': ['c_one']}, # 'theta': ['c_two']}, # 'theta': ['c_two']}, #regression_models = None, #
                                          regression_models = None, #regression_covariates = None,
                                          regression_covariates = None, # need this to make initial covariate matrix from which to use dmatrix (patsy)
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = None)
```

### Code cell 52

```python
# Make HDDM Model
hddmnn_model = hddm.HDDMnn(data,
                           model = model,
                           informative = False,
                           include = hddm.simulators.model_config[model]['hddm_include'],
                           p_outlier = 0.0,
                           is_group_model = True,
                           depends_on = {'v': 'c_one'})
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more than 1 hour**

### Code cell 54

```python
%%time
#Sample
hddmnn_model.sample(
    nmcmc,burn = 100,
    dbname='hddm_hier_by_con.db', db='pickle'
) 
hddmnn_model.save('hddm_hier_by_con')
```

### Code cell 55

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model,
                               ground_truth_parameter_dict = full_parameter_dict,
                               figsize = (8, 8),
                               columns = 3)

plt.show()
```

### Code cell 56

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model,
                                        columns = 2, # groupby = ['subj_idx'],
                                        figsize = (8, 6),
                                        value_range = np.arange(1, 2.5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_rts': True,
                                        'add_posterior_uncertainty_model': True,
                                        'samples': 200,
                                        'legend_fontsize': 7})
plt.show()
```

## Section 5: Regressors
This section provides a simple working example using the Neural Networks with the Regression backend. The regression back-end allows linking parameters to trial-by-trial covariates via a (general) linear model.

### Code cell 58

```python
# Metadata
nmcmc = 200
model = 'angle'
n_samples_by_subject = 500
```

### Code cell 59

```python
data, full_parameter_dict = simulator_h_c(n_subjects = 5,
                                          n_samples_by_subject = n_samples_by_subject,
                                          model = model,
                                          p_outlier = 0.00,
                                          conditions = None,
                                          depends_on = None,
                                          regression_models = ['t ~ 1 + covariate_name', 'v ~ 1 + covariate_name'],
                                          regression_covariates = {'covariate_name': {'type': 'continuous', 'range': (0, 1)}},
                                          group_only_regressors = False,
                                          group_only = None,
                                          fixed_at_default = None)
```

### Code cell 60

```python
# Set up the regressor a regressor:
reg_model_v = {'model': 'v ~ 1 + covariate_name', 'link_func': lambda x: x}
reg_model_t = {'model': 't ~ 1 + covariate_name', 'link_func': lambda x: x}
reg_descr = [reg_model_t, reg_model_v]
```

### Code cell 61

```python
# Make HDDM model
hddmnn_reg = hddm.HDDMnnRegressor(data,
                                  reg_descr,
                                  include = hddm.simulators.model_config[model]['hddm_include'],
                                  model = model,
                                  informative = False,
                                  p_outlier = 0.0)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more time**

### Code cell 63

```python
%%time
# Sample
hddmnn_reg.sample(
    nmcmc,burn = 100,
    dbname='hddm_reg.db', db='pickle'
) 
hddmnn_reg.save('hddm_reg')
```

### Code cell 64

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_reg,
                               ground_truth_parameter_dict = full_parameter_dict,
                               figsize = (8, 5),
                               columns = 3)

plt.show()
```

### Code cell 65

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_reg,
                                        columns = 2, # groupby = ['subj_idx'],
                                        figsize = (8, 6),
                                        value_range = np.arange(1, 2.5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_rts': True,
                                        'add_posterior_uncertainty_model': True,
                                        'samples': 200,
                                        'legend_fontsize': 7})
plt.show()
```

## Section 6: Stim Coding

### Code cell 67

```python
# Metadata
nmcmc = 300
model = 'ddm'
n_samples_by_condition = 500
split_param = 'v'
```

### Code cell 68

```python
sim_data_stimcoding, parameter_dict = hddm.simulators.simulator_stimcoding(model = model,
                                                                           split_by = split_param,
                                                                           drift_criterion = 0.3,
                                                                           n_trials_per_condition = 500)
```

### Code cell 69

```python
sim_data_stimcoding
```

### Code cell 70

```python
parameter_dict
```

### Code cell 71

```python
hddmnn_model = hddm.HDDMnnStimCoding(sim_data_stimcoding,
                                     include = hddm.simulators.model_config[model]['hddm_include'],
                                     model = model,
                                     stim_col = 'stim',
                                     p_outlier = 0.0,
                                     split_param = split_param,
                                     informative = False,
                                     drift_criterion = True)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more time**

### Code cell 73

```python
%%time
# Sample
hddmnn_model.sample(
    nmcmc,burn = 100,
    dbname='hddm_stim_code.db', db='pickle'
) 
hddmnn_model.save('hddm_stim_code')
```

### Code cell 74

```python
hddmnn_model.gen_stats()
```

### Code cell 75

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model,
                               ground_truth_parameter_dict = parameter_dict,
                               figsize = (8, 5),
                               columns = 3)

plt.show()
```

### Code cell 76

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_reg,
                                        columns = 2, # groupby = ['subj_idx'],
                                        figsize = (8, 6),
                                        value_range = np.arange(1, 2.5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_rts': True,
                                        'add_posterior_uncertainty_model': True,
                                        'samples': 200,
                                        'legend_fontsize': 7})
plt.show()
```

## Section 7: Model Recovery

### Code cell 78

```python
# note: here we merge the version of hddm by zenkavi, which repair the bug of fitting the weibull model 
model = 'weibull_cdf'
n_samples = 300
```

### Code cell 79

```python
# test regressors only False
# add p_outliers to the generator !
data, full_parameter_dict = hddm.simulators.hddm_dataset_generators.simulator_h_c(n_subjects = 1,
                                                                                  n_samples_by_subject = n_samples,
                                                                                  model = model,
                                                                                  p_outlier = 0.00,
                                                                                  conditions = None,
                                                                                  depends_on = None,
                                                                                  regression_models = None,
                                                                                  regression_covariates = None,
                                                                                  group_only_regressors = False,
                                                                                  group_only = None,
                                                                                  fixed_at_default = None)
```

### Code cell 80

```python
data
```

### Code cell 81

```python
# Now we fit for each model:
hddmnn_model_weibull = hddm.HDDMnn(data,
                                   informative = False,
                                   model = 'weibull_cdf',
                                   p_outlier = 0.0,
                                   include = hddm.simulators.model_config['weibull_cdf']['hddm_include'],
                                   is_group_model = False)

hddmnn_model_angle = hddm.HDDMnn(data,
                                 model = 'angle',
                                 informative = False,
                                 p_outlier = 0.0,
                                 include = hddm.simulators.model_config['angle']['hddm_include'],
                                 is_group_model = False)

hddmnn_model_ddm = hddm.HDDMnn(data,
                               informative = False,
                               model = 'ddm',
                               p_outlier = 0.0,
                               include = hddm.simulators.model_config['ddm']['hddm_include'],
                               is_group_model = False)
```

### Code cell 82

```python
%%time
# Sample
nmcmc = 500
hddmnn_model_weibull.sample(
    nmcmc,
    burn = 200,
    dbname='hddm_weibull.db', db='pickle'
)
hddmnn_model_weibull.save('hddm_weibull')

hddmnn_model_angle.sample(
    nmcmc,
    burn = 200,
    dbname='hddm_angle.db', db='pickle'
)
hddmnn_model_angle.save('hddm_angle')

hddmnn_model_ddm.sample(
    nmcmc,
    burn = 200,
    dbname='hddm_ddm.db', db='pickle'
)
hddmnn_model_ddm.save('hddm_ddm')
```

### 7.1 Checking Model Fits Visually
Posterior Predictive: Do the ‘Posterior Models’ also make sense?

### Code cell 84

```python
# WEIBULL
hddm.plotting.plot_posterior_predictive(model = hddmnn_model_weibull,
                                        columns = 1,
                                        groupby = ['subj_idx'],
                                        figsize = (6, 6),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = True,
                                        **{'alpha': 0.01,
                                        'ylim': 5,
                                        'add_posterior_uncertainty_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_mean_rts': True,
                                        'samples': 200})
plt.show()
```

### Code cell 85

```python
# ANGLE
hddm.plotting.plot_posterior_predictive(model = hddmnn_model_angle,
                                        columns = 1,
                                        groupby = ['subj_idx'],
                                        figsize = (6, 6),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = False,
                                        **{'alpha': 0.01,
                                        'ylim': 5,
                                        'add_posterior_uncertainty_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_mean_rts': True,
                                        'samples': 200})
plt.show()
```

### Code cell 86

```python
# DDM
hddm.plotting.plot_posterior_predictive(model = hddmnn_model_ddm,
                                        columns = 1,
                                        groupby = ['subj_idx'],
                                        figsize = (6, 6),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = False,
                                        **{'alpha': 0.01,
                                        'ylim': 5,
                                        'add_posterior_uncertainty_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_mean_rts': True,
                                        'samples': 200})
plt.show()
```

### 7.2 Comparing DIC’s

### Code cell 88

```python
hddmnn_model_weibull.dic
```

### Code cell 89

```python
hddmnn_model_angle.dic
```

### Code cell 90

```python
hddmnn_model_ddm.dic
```

## Section 8: Real Data!

### Code cell 92

```python
# Metadata
nmcmc = 1000
burn = 500
model = 'angle'
```

### 8.1 Load and Pre-process dataset

### Code cell 94

```python
# Load one of the datasets shipping with HDDM
cav_data = hddm.load_csv(hddm.__path__[0] + '/examples/cavanagh_theta_nn.csv')
```

### Code cell 95

```python
cav_data
```

### 8.2 Basic Condition Split Model

### Code cell 97

```python
hddmnn_model_cav = hddm.HDDMnn(cav_data,
                               model = model,
                               informative = False,
                               include = hddm.simulators.model_config[model]['hddm_include'],
                               p_outlier = 0.05,
                               is_group_model = False,
                               depends_on = {'v': 'stim'})
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more time**

### Code cell 99

```python
%%time
# Sample
hddmnn_model_cav.sample(
    nmcmc,burn = burn,
    dbname='hddm_model_cav1.db', db='pickle'
) 
hddmnn_model_cav.save('hddm_model_cav1')
```

### Code cell 100

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model_cav,
                                        columns = 1,
                                        figsize = (6, 6),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = False,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_mean_rts': True,
                                        'samples': 200})
plt.show()
```

### 8.3 Basic Hierarchical Model

### Code cell 102

```python
hddmnn_model_cav = hddm.HDDMnn(cav_data,
                               model = model,
                               informative = False,
                               include = hddm.simulators.model_config[model]['hddm_include'],
                               is_group_model = True,
                               p_outlier = 0.05)
```

**Note: If yhddmnn_modelour cpu run low on computing, the following code may take more than 50 minutes**

### Code cell 104

```python
%%time
# Sample
hddmnn_model_cav.sample(
    nmcmc,burn = burn,
    dbname='hddm_model_cav2.db', db='pickle'
) 
hddmnn_model_cav.save('hddm_model_cav2')
```

### Code cell 105

```python
# Caterpillar Plot: (Parameters recovered ok?)
hddm.plotting.plot_caterpillar(hddm_model = hddmnn_model_cav,
                               figsize = (8, 8),
                               columns = 3)

plt.show()
```

### Code cell 106

```python
hddm.plotting.plot_posterior_predictive(model = hddmnn_model_cav,
                                        columns = 3,
                                        figsize = (10, 10),
                                        value_range = np.arange(0, 5, 0.1),
                                        plot_func = hddm.plotting._plot_func_model,
                                        parameter_recovery_mode = False,
                                        **{'alpha': 0.01,
                                        'ylim': 3,
                                        'add_posterior_uncertainty_model': True,
                                        'add_posterior_uncertainty_rts': False,
                                        'add_posterior_mean_rts': True,
                                        'samples': 200,
                                        'legend_fontsize': 7,
                                        'subplots_adjust': {'top': 0.9, 'hspace': 0.3, 'wspace': 0.3}})
plt.show()
```

### Note
This is just an example. The angle model might not be the best choice here, and we are moreover ignoring the supplied conditions.

## Section 9: Accessing the Neural Network Directly

### 9.1 Direct access to batch predictions

### Code cell 110

```python
model = 'angle'
```

### Code cell 111

```python
lan_angle = hddm.network_inspectors.get_torch_mlp(model = model)
```

Let’s predict some likelihoods !

### Code cell 113

```python
# Make some random parameter set
parameter_df = hddm.simulators.make_parameter_vectors_nn(model = model,
                                                         param_dict = None,
                                                         n_parameter_vectors = 1)
parameter_df
```

### Code cell 114

```python
parameter_matrix = np.tile(np.squeeze(parameter_df.values), (200, 1))
parameter_matrix[1:10]
```

### Code cell 115

```python
# Initialize network input
network_input = np.zeros((parameter_matrix.shape[0], parameter_matrix.shape[1] + 2)) # Note the + 2 on the right --> we append the parameter vectors with reaction times (+1 columns) and choices (+1 columns)
network_input
```

### Code cell 116

```python
# Add reaction times
network_input[:, -2] = np.linspace(0, 3, parameter_matrix.shape[0])
network_input
```

### Code cell 117

```python
# Add choices
network_input[:, -1] = np.repeat(np.random.choice([-1, 1]), parameter_matrix.shape[0])
network_input
```

### Code cell 118

```python
# Convert to float
network_input = network_input.astype(np.float32)
# Show example output
print(lan_angle(network_input)[:10]) # printing the first 10 outputs
print(lan_angle(network_input).shape) # original shape of output
```

### Code cell 119

```python
network_input[:,0:5] = parameter_matrix
network_input
```

### Code cell 120

```python
lan_angle(network_input)[1:10]
```

### 9.2 Plotting Utilities

HDDM provides two plotting function to investigate the network outputs directly. The `kde_vs_lan_likelihoods()` plot and the `lan_manifold()` plot.

#### 9.2.1 `kde_vs_lan_likelihoods()`
The `kde_vs_lan_likelihoods()` plot allows you to check the likelihoods produced by a LAN against Kernel Density Estimates (KDEs) from model simulations. You can supply a panda `DataFrame` that holds parameter vectors as rows.

### Code cell 123

```python
# Make some parameters
parameter_df = hddm.simulators.make_parameter_vectors_nn(model = model,
                                                         param_dict = None,
                                                         n_parameter_vectors = 10)
```

### Code cell 124

```python
parameter_df
```

### Code cell 125

```python
hddm.network_inspectors.kde_vs_lan_likelihoods(parameter_df = parameter_df,
                                               model = model,
                                               cols = 3,
                                               n_samples = 2000,
                                               n_reps = 10,
                                               show = True)
```

#### 9.2.2 `lan_manifold()`

### Code cell 127

```python
# Make some parameters
parameter_df = hddm.simulators.make_parameter_vectors_nn(model = model,
                                                         param_dict = None,
                                                         n_parameter_vectors = 1)
```

### Code cell 128

```python
parameter_df
```

### Code cell 129

```python
# Now plotting
hddm.network_inspectors.lan_manifold(parameter_df = parameter_df,
                                     vary_dict = {'v': np.linspace(-2, 2, 20)},
                                     model = model,
                                     n_rt_steps = 300,
                                     fig_scale = 1.0,
                                     max_rt = 5,
                                     save = True,
                                     show = True)
```
