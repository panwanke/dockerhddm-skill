# dockerHDDM Workflow

> Converted from `dockerHDDM_Workflow.ipynb`. Code is preserved; rich outputs are omitted.

# <a id='toc1_'></a>[Example workflow in dockerHDDM](#toc0_)

Author: 

- Wanke Pan (<EMAIL>) @Nanjing Normal University
- Hu Chuan-Peng (corresponding author, <EMAIL>) @Nanjing Normal University
- Ru-Yuan Zhang (corresponding author, <EMAIL>) @Shanghai Jiao Tong University

This is a supplementary notebook for the introductary paper *dockerHDDM: A user-friendly environment for Bayesian Hierarchical Drift-Diffusion Modeling.*. 

This notebook was tested in docker image [`hcp4715/hddm:1.0.1`](https://hub.docker.com/r/hcp4715/hddm/tags), where the packages `kabuki` were rectfied to RC version as below code box.

Preprint of this manuscript: https://psyarxiv.com/6uzga/

Github repository at: https://github.com/hcp4715/dockerHDDM

**Table of contents**<a id='toc0_'></a>    
- [Example workflow in dockerHDDM](#toc1_)    
  - [Loading modules/packages](#toc1_1_)    
  - [Load the example data Cavanaght et al. (2011)](#toc1_2_)    
  - [Model specification and fitting](#toc1_3_)    
  - [Model diagnosis](#toc1_4_)    
    - [Diagnosis of Model 0](#toc1_4_1_)    
      - [Trace plot](#toc1_4_1_1_)    
      - [$\hat{R}$ and ESS](#toc1_4_1_2_)    
    - [Diagnosis of Model 1](#toc1_4_2_)    
      - [Trace plot](#toc1_4_2_1_)    
      - [$\hat{R}$ and ESS](#toc1_4_2_2_)    
    - [Diagnosis of Model 2](#toc1_4_3_)    
      - [Trace plot](#toc1_4_3_1_)    
      - [$\hat{R}$ and ESS](#toc1_4_3_2_)    
      - [non-converge example](#toc1_4_3_3_)    
  - [Model comparison and selection](#toc1_5_)    
    - [DIC](#toc1_5_1_)    
    - [PSIS-LOO-CV](#toc1_5_2_)    
      - [ WAIC](#toc1_5_2_1_)    
      - [PSIS-LOO-CV](#toc1_5_2_2_)    
    - [PSIS issues](#toc1_5_3_)    
  - [Posterior predictive check](#toc1_6_)    
    - [Custom PPC plotting function](#toc1_6_1_)    
  - [Statistical Inference](#toc1_7_)    

<!-- vscode-jupyter-toc-config
	numbering=false
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

## <a id='toc1_1_'></a>[Loading modules/packages](#toc0_)

### Code cell 4

```python
%matplotlib inline

# scitnific computing and plotting
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# HDDM related packages
import pymc as pm
import hddm
import kabuki
import arviz as az
print("The current HDDM version is: ", hddm.__version__)
print("The current kabuki version is: ", kabuki.__version__)
print("The current PyMC version is: ", pm.__version__)
print("The current ArviZ version is: ", az.__version__)
```

## <a id='toc1_2_'></a>[Load the example data Cavanaght et al. (2011)](#toc0_)

The data file is included in HDDM, so we directly load it.

### Code cell 6

```python
data_cavanagh = hddm.load_csv(hddm.__path__[0] + '/examples/cavanagh_theta_nn.csv')
data_cavanagh.tail()
```

### Code cell 7

```python
print("The number of trials: ", data_cavanagh.shape[0])
print("The number of variables: ", data_cavanagh.shape[1])
print("The number of participants: ", data_cavanagh.subj_idx.unique().shape[0])
```

## <a id='toc1_3_'></a>[Model specification and fitting](#toc0_)

We define 3 models as below table (Table 2 in main text) from the simplest DDM to full DDM to regression models. Explanation for these three models can be found in **Section "Data and Example Models"** in the main text. 

| Models | HDDM functions for defining a model (`df` is the data from Cavanagh et al., 2011) | # param |
|-|-|-|  
| m0 | hddm.HDDM(df, include=\'a', 'v', 't', *'z', 'sv', 'sz', 'st\]) | 67 |
| m1 | hddm.HDDM(df, include=\['a', 'v', 't','z', 'sv', 'st', 'sz'\], **depends\_on={'v': 'conf'}**) | 82 |
| m2 | hddm.HDDMRegressor(df, **"v ~ C(conf, Treatment('LC'))"**, **group\_only\_regressors=False**, keep\_regressor\_trace=True, include=\['a', 'v', 't', 'z', 'sv', 'st', 'sz'\]) | 83 |

### Code cell 10

```python
df = data_cavanagh.copy()

# Model 0: base model: full model
m0 = hddm.HDDM(df, include=['a', 'v', 't', 'z', 'sv', 'sz', 'st'])

# Model 1: treat within-subj as between-subj: full model
m1 = hddm.HDDM(df, include=['a', 'v', 't','z', 'sv', 'st', 'sz'], depends_on={'v': 'conf'})

# Model 2: regression model (varying intercept and slope)
m2 = hddm.HDDMRegressor(
  df, "v ~ 1 + C(conf, Treatment('LC'))", 
  include=['a', 'v', 't', 'z', 'sv', 'st', 'sz'], 
  group_only_regressors=False, 
  keep_regressor_trace=True)
```

Fitting model with new features (parallel sampling and return inferenceData, see [dockerHDDM Quick View](./dockerHDDM_Quick_View.ipynb) and **Section New features in dockerHDDM**). 

We set up 4 MCMC chains with 10,000 samples with 5,000 burn-ins with parallel fitting process and return inferred data for by the code `model.sample(10000, burn = 5000, return_infdata = True)`.

### Code cell 12

```python
def run_sampling(
    m,
    n_samples=10000,
    n_burn=5000,
    n_chains=4,
    model_name="m",
    progress_bar=True,
):
    """
    Run Markov Chain Monte Carlo (MCMC) sampling for a given model and parameters.
    This function takes a model parameter `m` and runs MCMC sampling using PyMC2.3.8.
    It returns a arviz InferenceData object containing the posterior samples and post-fit model.
    Parameters
    ----------
    m : object
        The model parameter to be sampled.
    n_samples : int, optional
        The number of posterior samples to generate. Default is 10000.
    n_burn : int, optional
        The number of initial samples to discard. Default is 5000.
    n_chains : int, optional
        The number of MCMC chains to run in parallel. Default is 4.
    model_name : str, optional
        The name of the model to be used in the sampling process. Default is "m".
    progress_bar : boolean, optional
        Default=True. Set False hide sampling progress bar. 
    Returns
    -------
    InferenceData : arviz.InferenceData
        A trace object containing the posterior samples.
    Model: HDDM
        A post-fit HDDM model. 
    """

    m_infdata = m.sample(n_samples,
                         burn=n_burn,
                         chains=n_chains,
                         return_infdata=True,
                         save_name="model_fitted/" + model_name,
                         progress_bar=progress_bar)

    return m, m_infdata
```

**Note: Fitting all these six model with settings above (10000 samples) takes about 8-9 hours on a PC with Intel® Core™ i7-10700 CPU @ 2.90GHz. Also, the memory of should be large enough, ~ 64 G memory is recommended. Otherwise,  recommend few samples (e.g., n_samples=2500, burn=1000).**

### Code cell 14

```python
%time
m0, m0_infdata = run_sampling(m0, model_name="m0")
```

### Code cell 15

```python
%time
# Set progress_bar=False to hide the program's progress bar to simplify the output
m1, m1_infdata = run_sampling(m1, model_name="m1")
```

### Code cell 16

```python
%time
# Set progress_bar=False to hide the program's progress bar to simplify the output
m2, m2_infdata = run_sampling(m2, model_name="m2")
```

## <a id='toc1_4_'></a>[Model diagnosis](#toc0_)

We use `az.plot_trace()` to visually check the trace plot of MCMC chains and use `az.summary()` to check $\hat{R}$ and Effective Sample Size (ESS).

### <a id='toc1_4_1_'></a>[Diagnosis of Model 0](#toc0_)

#### <a id='toc1_4_1_1_'></a>[Trace plot](#toc0_)

We can also only plot selected parameters' traces. 
 
- use regex to select var_names that start with "a" and do not contain either "subj" or "std".

### Code cell 22

```python
tmp_infdata = m0_infdata
plt.rc('font', size=16)

# plot traces for all parameters
axes = az.plot_trace(tmp_infdata)
plt.gcf().set_size_inches(7, 70)
```

### Code cell 23

```python
# select population level parameter v as a example
axes = az.plot_trace(
    tmp_infdata,
    var_names='v'
)
plt.gcf().set_size_inches(10, 4)
```

### Code cell 24

```python
# select all population level parameters
axes = az.plot_trace(
    tmp_infdata,
    var_names=['~subj'],  # exclude individual level parameters
    filter_vars='regex'
)
plt.gcf().set_size_inches(7, 20)
```

#### <a id='toc1_4_1_2_'></a>[$\hat{R}$ and ESS](#toc0_)

### Code cell 26

```python
summary_tmp = az.summary(tmp_infdata, kind = "diagnostics", round_to=4)
summary_tmp.sort_values('r_hat', ascending=False).head(10)
```

### Code cell 27

```python
# only calculate R hat
az.rhat(tmp_infdata).values
```

### Code cell 28

```python
# only calculate ESS
az.ess(tmp_infdata).values
```

### <a id='toc1_4_2_'></a>[Diagnosis of Model 1](#toc0_)

#### <a id='toc1_4_2_1_'></a>[Trace plot](#toc0_)

### Code cell 31

```python
tmp_infdata = m1_infdata

# plot traces for all parameters
# axes = az.plot_trace(tmp_infdata)
# plt.gcf().set_size_inches(7, 70)
```

### Code cell 32

```python
# select all population level parameters
axes = az.plot_trace(
    tmp_infdata,
    var_names=['~subj'],  # exclude individual level parameters
    filter_vars='regex'
)
plt.gcf().set_size_inches(7, 20)
```

#### <a id='toc1_4_2_2_'></a>[$\hat{R}$ and ESS](#toc0_)

### Code cell 34

```python
summary_tmp = az.summary(tmp_infdata, kind = "diagnostics", round_to=4)
summary_tmp.sort_values('r_hat', ascending=False).head(10)
```

### <a id='toc1_4_3_'></a>[Diagnosis of Model 2](#toc0_)

#### <a id='toc1_4_3_1_'></a>[Trace plot](#toc0_)

### Code cell 37

```python
tmp_infdata = m2_infdata

# plot traces for all parameters
# axes = az.plot_trace(tmp_infdata)
# plt.gcf().set_size_inches(7, 70)
```

### Code cell 38

```python
# select all population level parameters
axes = az.plot_trace(
    tmp_infdata,
    var_names=['~subj'],  # exclude individual level parameters
    filter_vars='regex'
)
plt.gcf().set_size_inches(7, 40)
```

#### <a id='toc1_4_3_2_'></a>[$\hat{R}$ and ESS](#toc0_)

### Code cell 40

```python
summary_tmp = az.summary(tmp_infdata, kind = "diagnostics", round_to=4)
summary_tmp.sort_values('r_hat', ascending=False).head(10)
```

#### <a id='toc1_4_3_3_'></a>[non-converge example](#toc0_)

Since `z_std` have $\hat{R}$ > 1.01 and ESS bulk < 400, we double-check these two parameters with a trace plot.

- In terms of the posterior distribution, 'z_std' differ in the consistency of convergence of the four chains and their hdi (highest density interval).
- Since the 'z_std' obeys the truncate distribution, its mean and mode are clustered around 0
- However, from the trace plot, it is clear that 'z_std' has non-converge feature in chain 2 (the long regions of monotonicity like a line parallel to the x-axis).
    - (see 2.4 Diagnosing Numerical Inference in Martin, O.A., et. cl (2021) Bayseian Modeling and Computation in Python).

However, in general, the $\hat{R}$ of the two parameters is close to 1.01, and more broadly speaking, less than 1.1. And the distribution of the chains is still stable and consistent, we think this result is acceptable.

### Code cell 43

```python
# select `z_std`
axes = az.plot_trace(
    m2_infdata,
    var_names=['z_std']
)
plt.gcf().set_size_inches(10, 4)
```

### Code cell 44

```python
for i in range(4):
    axes = az.plot_trace(
        m2_infdata,
        var_names=['z_std'],
        coords={'chain': [i]}
    )
```

## <a id='toc1_5_'></a>[Model comparison and selection](#toc0_)

### <a id='toc1_5_1_'></a>[DIC](#toc0_)

Here we retrieve the DIC from each model in the models and sort by DIC.

### Code cell 47

```python
dic_dict = {
  "m0(baseline)":m0.dic,
  "m1(v depends on conf)":m1.dic,
  "m2(reg: v ~ 1 + conf)":m2.dic
}

comp_dic = pd.DataFrame.from_dict(dic_dict, orient='index', columns=['DIC'])
comp_dic['model'] = comp_dic.index
comp_dic = comp_dic[['model', 'DIC']]
comp_dic.sort_values(by=['DIC'], ascending = True)
```

### <a id='toc1_5_2_'></a>[PSIS-LOO-CV](#toc0_)

To evaluate models with criterions like WAIC and PSIS-LOO-CV, pointwise loglikelihoods are required (see Section Model Comparison in Manuscript). 

The following code shows how to calculate the pointwise loglikelihoods. 

**note**:
- **It can take half an hour or a few hours depending on CPU performance**.
- **If your RAM is less than or equal to 16g, an error may occur**.

### Code cell 50

```python
%time
m0_infdata = m0.to_infdata(loglike = True, save_name = "model_fitted/m0")
```

### Code cell 51

```python
%time
m1_infdata = m1.to_infdata(loglike = True, save_name = "model_fitted/m1")
```

### Code cell 52

```python
%time
m2_infdata = m2.to_infdata(loglike = True, save_name = "model_fitted/m2")
```

Warining here is caused by large number as input of `exp()`, solution is [here](https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function).

### Code cell 54

```python
compare_dict = {
  "m0":m0_infdata,
  "m1":m1_infdata,
  "m2":m2_infdata
}
```

#### <a id='toc1_5_2_1_'></a>[ WAIC](#toc0_)

### Code cell 56

```python
comp_waic = az.compare(compare_dict, var_name='log_lik', round_to='none')
comp_waic
```

#### <a id='toc1_5_2_2_'></a>[PSIS-LOO-CV](#toc0_)

### Code cell 58

```python
comp_loo = az.compare(compare_dict, var_name='log_lik', round_to='none')
comp_loo
```

### <a id='toc1_5_3_'></a>[PSIS issues](#toc0_)

**Warnings**: The above result shows that m0 and m1 are unable to compute PSIS-LOO-CV(elpd_loo), this is because of the overflow encountered in some calculation as the above warnings.

As follow, we can check pareto_k $\hat{k}$. If $\hat{k} >  0.7$, which means importance sampling (PSIS) is not able to provide useful estimate for that component/observation. Highly influential observations have high $\hat{k}$ values. Very high $\hat{k}$ values often indicate model misspecification, outliers or mistakes in data processing.

Also, `p_loo`is large than *p*, which is number of parameters of the model. Large `p_loo` means the model is badly misspecified according to Aki Vehtari's [Q & A](https://avehtari.github.io/modelselection/CV-FAQ.html#18_What_is_the_interpretation_of_p_loo). For all models, the *N*, number of obersvations, is 3988, and the number of parameters varies from 48 to 83. Thus, `p_loo` is much larger than numbers of parameters for all models.

As we can from below, this was primarily because of a few outliers. As HDDM allow for 5 percent of outliers in modelling, it is not surprising that these outliers stronly influenced the model performance. Here, we remove the outliers from all models and check the results again.

Below we visulaize the `k_hat` of each data point.

### Code cell 62

```python
%time
loo_m1 = az.loo(m1_infdata, var_name='log_lik', pointwise=True)
az.plot_khat(loo_m1, threshold=0.7)
```

Plot pointwise elpd differences between two or more models

### Code cell 64

```python
%time
az.plot_elpd({"m0": m0_infdata, "m1": m1_infdata}, xlabels=True)
```

Here we used a very simple soluation: removing the outliers and re-calculate the `loo`.

### Code cell 66

```python
%time
loo_m0 = az.loo(m0_infdata, var_name='log_lik', pointwise=True)
loo_m1 = az.loo(m1_infdata, var_name='log_lik', pointwise=True)
loo_m2 = az.loo(m2_infdata, var_name='log_lik', pointwise=True)
```

### Code cell 67

```python
outliers0 = loo_m0.pareto_k.where(loo_m0.pareto_k >= 0.7, drop= True).obs_id.values
outliers1 = loo_m1.pareto_k.where(loo_m1.pareto_k >= 0.7, drop= True).obs_id.values
outliers2 = loo_m2.pareto_k.where(loo_m2.pareto_k >= 0.7, drop= True).obs_id.values

# create a index array to remove outliers
outliers = np.unique(np.concatenate((outliers0, outliers1, outliers2), axis=0))
new_indx = data_cavanagh.index.values[~np.isin(data_cavanagh.index.values, outliers)]
```

### Code cell 68

```python
%time
m0_infdata = m0_infdata.isel(obs_id=new_indx)
m1_infdata = m1_infdata.isel(obs_id=new_indx)
m2_infdata = m2_infdata.isel(obs_id=new_indx)
```

### Code cell 69

```python
compare_dict2 = {
  "m0(delete outliers)":m0_infdata,
  "m1(delete outliers)":m1_infdata,
  "m2(delete outliers)":m2_infdata
}
```

### Code cell 70

```python
comp_loo2 = az.compare(compare_dict2, var_name='log_lik', round_to='none')
comp_loo2
```

## <a id='toc1_6_'></a>[Posterior predictive check](#toc0_)

First, let's spend some time generating posterior predictions. Note that we set `n_ppc` is 500 means we generate 500 samples for each draws in each parameters.

### Code cell 73

```python
%time
m0_infdata = m0.to_infdata(loglike = True, ppc = True, n_ppc = 500, save_name = "model_fitted/m0")
m1_infdata = m1.to_infdata(loglike = True, ppc = True, n_ppc = 500, save_name = "model_fitted/m1")
m2_infdata = m2.to_infdata(loglike = True, ppc = True, n_ppc = 500, save_name = "model_fitted/m2")
```

Then, we can plot the ppc by setting different coordinates. 

- Here we demonstrate plotting both the individual leve, `subj_idx` is 3, 11
- and the experimental condition level, `conf` is LC and HC.

### Code cell 75

```python
# ArviZ 1.1 uses DataTree-based plotting. Keep the PPC plot simple and
# compatible, then use custom helpers below for subject/condition-specific PPC.
axes = az.plot_ppc_dist(
    m0_infdata,
    var_names='rt',
    num_samples=100
)
plt.gcf().set_size_inches(8, 5)
```

### Code cell 76

```python
# render figure for manuscript
plt.gcf().savefig('fig_ppc_m0_by_subject.pdf')
```

### Code cell 77

```python
axes = az.plot_ppc_dist(
    m2_infdata,
    var_names='rt',
    num_samples=100
)
plt.gcf().set_size_inches(8, 5)
```

### Code cell 78

```python
# render figure for manuscript
plt.gcf().savefig('fig_ppc_m2_by_subject.pdf')
```

It is found that model 2 generate more similar predictions to observed data, especially for subject 11. 

Then, let us plot ppc on different conditions.

### Code cell 80

```python
# PPC distribution for model 0. For condition-specific views, use the
# kabuki.plot_ppc_by_cond helper demonstrated below.
axes = az.plot_ppc_dist(
    m0_infdata,
    var_names='rt',
    num_samples=100
)
plt.gcf().set_size_inches(8, 5)
```

### Code cell 81

```python
# render figure for manuscript
plt.gcf().savefig('fig_ppc_m0_by_condition.pdf')
```

### Code cell 82

```python
# PPC distribution for model 2. For condition-specific views, use the
# kabuki.plot_ppc_by_cond helper demonstrated below.
axes = az.plot_ppc_dist(
    m2_infdata,
    var_names='rt',
    num_samples=100
)
plt.gcf().set_size_inches(8, 5)
```

### Code cell 83

```python
# render figure for manuscript
plt.gcf().savefig('fig_ppc_m2_by_condition.pdf')
```

model 2 get better predictions for both conflict condition (LC vs. HC), given that model 0 did not consider the impact from conflict condition.

### <a id='toc1_6_1_'></a>[Custom PPC plotting function](#toc0_)

To make it easier to plot ppc under different conditions, we define a function plot_ppc_by_cond, demonstrated as follows.

### Code cell 86

```python
from kabuki.analyze import plot_ppc_by_cond

# Examples:
# plot_ppc_by_cond(m2_infdata, num_pp_samples=100, seed=2024)
# plot_ppc_by_cond(m2_infdata, subj_idx=[0, 3], num_pp_samples=100, seed=2024)
# plot_ppc_by_cond(m2_infdata, condition_vars='conf', num_pp_samples=100, seed=2024)
# plot_ppc_by_cond(m2_infdata, condition_vars=['conf', 'stim'], num_pp_samples=100, seed=2024)
# plot_ppc_by_cond(m2_infdata, condition_vars={'stim': ['LL', 'WW']}, num_pp_samples=100, seed=2024)

condition_vars = ['stim', 'conf']
subj_idx = [0, 3]
axes = plot_ppc_by_cond(
    m2_infdata,
    subj_idx=subj_idx,
    condition_vars=condition_vars,
    num_pp_samples=100,
    seed=2024,
    legend=False,
    textsize=12,
    alpha=0.35,
    max_cols=4,
)
plt.show()
```

## <a id='toc1_7_'></a>[Statistical Inference](#toc0_)

Here we onlyu examplified how to us ROPE + HDI method. 

- Assume that we are interested in the effect of conflit level on drift rate `v` and we used [-0.2, 0.2] as the ROPE. 
- Then we can used visualize the ROPE and HDI with `az.plot_posterior()`.

### Code cell 88

```python
# Visualize the posterior difference with a manual histogram because
# the legacy posterior helper is not available in ArviZ 1.1.
var_name = "v_C(conf, Treatment('LC'))[T.HC]"
samples = m2_infdata.posterior[var_name].values.reshape(-1)
hdi = az.hdi(samples, hdi_prob=0.95)
fig, axes = plt.subplots(figsize=(5, 5))
axes.hist(samples, bins=40, density=True, alpha=0.75, color='C0')
axes.axvspan(-0.2, 0.2, color='r', alpha=0.15, label='ROPE')
axes.axvline(hdi[0], color='k', linestyle='--', linewidth=1, label='95% HDI')
axes.axvline(hdi[1], color='k', linestyle='--', linewidth=1)
axes.axvline(0, color='0.35', linewidth=1)
axes.set_title(var_name)
axes.set_xlim(-0.9, 0.3)
axes.legend()
```

### Code cell 89

```python
# render figure for manuscript
axes.figure.savefig("fig_posterior_inference_m2_drift_rate_on_conflict.pdf")
```

It is easy to find that the drift rate in HC is lower than it in LC. 

Then we plot the posteriors distribution of drift rates in both condition to reaffirm this results.

### Code cell 91

```python
# Plot posterior histograms by condition. v_Intercept is v in LC and
# v_C(conf, Treatment('LC'))[T.HC] is the HC-LC contrast.
lc = m2_infdata.posterior['v_Intercept'].values.reshape(-1)
contrast = m2_infdata.posterior["v_C(conf, Treatment('LC'))[T.HC]"].values.reshape(-1)
hc = lc + contrast
fig, axes = plt.subplots(figsize=(10, 5))
axes.hist(lc, bins=40, density=True, alpha=0.55, label='v_LC')
axes.hist(hc, bins=40, density=True, alpha=0.55, label='v_HC')
axes.set_xlabel('Drift rate')
axes.set_ylabel('Density')
axes.legend()
```

### Code cell 92

```python
m2_infdata.posterior["v_HC"] = (
    ("chain", "draw"),
    m2_infdata.posterior["v_Intercept"].values + \
    m2_infdata.posterior["v_C(conf, Treatment('LC'))[T.HC]"].values)
m2_infdata.posterior["v_LC"] = (
    ("chain", "draw"),
    m2_infdata.posterior["v_Intercept"].values)
```

Compare posterior using plot_forest. See [`az.plot_forest`](https://python.arviz.org/en/latest/api/generated/arviz.plot_forest.html#arviz.plot_forest) for more.

### Code cell 94

```python
# Plot posterior intervals by condition using ArviZ 1.1's forest plot API.
axes = az.plot_forest(
    m2_infdata,
    var_names=['v_LC', 'v_HC'],
    combined=True,
    point_estimate='median',
    ci_probs=[0.5, 0.95]
)
plt.gcf().set_size_inches(8, 5)
```

Similarly, violin plot can also be used:

### Code cell 96

```python
import seaborn as sns

posterior_df = pd.DataFrame({
    'v_LC': m2_infdata.posterior['v_LC'].values.reshape(-1),
    'v_HC': m2_infdata.posterior['v_HC'].values.reshape(-1),
}).melt(var_name='condition', value_name='drift_rate')

fig, axes = plt.subplots(figsize=(6, 5))
sns.violinplot(data=posterior_df, x='condition', y='drift_rate', ax=axes, inner='quartile')
sns.despine(ax=axes)
```

### Code cell 97

```python
# render figure for manuscript
plt.gcf().savefig('fig_posterior_inference_m2_drift_rate_on_conflict_violin.pdf')
```

If we have prior samples in inferenceData especially when we set `sample_prior = True` in `to_infdata` or `sample` methods. We could use `az.plot_bf` to calculate Savage-Dickey Density Ratio to approximate the Bayes Factor.

### Code cell 99

```python
m2_infdata = m2.to_infdata(sample_prior=True, save_name = "model_fitted/m2")
```

### Code cell 100

```python
import matplotlib.pyplot as plt
ax = az.plot_bf(
    m2_infdata,
    var_names=["v_C(conf, Treatment('LC'))[T.HC]"],
    ref_val=0
)
plt.xlim(-1, 0.5)
```

### Code cell 101

```python
# render figure for manuscript
plt.gcf().savefig('fig_posterior_inference_m2_drift_rate_on_conflict_bf.pdf')
```

### Code cell 102

```python
import matplotlib.pyplot as plt
ax = az.plot_bf(
    m2_infdata,
    var_names=['z'],
    ref_val=0.5
)
plt.xlim(0.3, 0.7)
```

### Code cell 103

```python
# render figure for manuscript
plt.gcf().savefig('fig_posterior_inference_m2_z_bf.pdf')
```
