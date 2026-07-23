# dockerHDDM Quick View

> Converted from `dockerHDDM_Quick_View.ipynb`. Code is preserved; rich outputs are omitted.

# <a id='toc1_'></a>[Ten Minutes Testing New Features](#toc0_)

> Integrating Multi-Chain MCMC and Arviz Functionality into the HDDM

As is well known, the HDDM itself does not support fitting multiple chains in MCMC, so fitting an HDDM model with multiple chains requires users to perform complex customization. In addition, the Bayesian plotting and testing capabilities included in HDDM are relatively rudimentary, and leveraging arviz (based on xarray's InferenceData) can help users perform more flexible post-analysis.

Previous work in our laboratory, [dockerHDDM](https://github.com/hcp4715/dockerHDDM), has implemented multiple-chain MCMC fitting and conversion of hddm models to arviz InferenceData. However, these methods have not been well integrated into the modules on which hddm itself depends. This work ported these features to the core dependency library of hddm, `kabuki`, which fundamentally solves the problems that existed before and makes it more convenient for users to use.

We hope that this notebook can provides a good example of how to use our advanced update to run multiple chains in parallel and generate an ArViz InferenceData from the hddm class.

Firstly, ensure that you are operating within the dockerHDDM environment. Should you find yourself in a local Python or Conda environment rather than the dockerHDDM, which comes pre-installed with the official HDDM, please proceed to remove the original Kabuki package and install our development version from github at https://github.com/panwanke/kabuki, which is compatible with hddm >= 0.8.0.
To ensure that all functions properly, we recommend using our HDDM development version at https://github.com/panwanke/hddm. 


```
%pip uninstall kabuki -y
%pip install git+https://github.com/panwanke/kabuki
%pip uninstall hddm -y
%pip install git+https://github.com/panwanke/hddm
```

And you could see the development version is kabuki 0.6.5RC3 and hddm 0.9.8RC.

**Table of contents**<a id='toc0_'></a>    
- [Ten Minutes Testing New Features](#toc1_)    
    - [Basic Bayesian hierarchical model](#toc1_1_1_)    
    - [HDDM with regressors](#toc1_1_2_)    
    - [model comparison with LOO](#toc1_1_3_)    
  - [statistical inference](#toc1_2_)    

<!-- vscode-jupyter-toc-config
	numbering=false
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

### Code cell 4

```python
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import hddm
import kabuki
import arviz as az

print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
print("The current version of arviz is: ", az.__version__)
```

Loading the data from the hddm package

### Code cell 6

```python
data = hddm.load_csv(hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv")
data = data[data['subj_idx'].isin([0,1,2,3,4])]
data.head()
```

Sampling a basic model to ensure that the python enviroment is working properly.

### Code cell 8

```python
model = hddm.HDDM(data, include = ['v', 'a', 't', 'z'])
model.sample(100)
```

### <a id='toc1_1_1_'></a>[Basic Bayesian hierarchical model](#toc0_)

Now, let we **run the model with 4 chains in parallel**, and **return the ArViz InferenceData**.

### Code cell 11

```python
%time
model0 = hddm.HDDM(data, include = ['v', 'a', 't', 'z'])
# note: when chains more than one, sampling process will be run parallelly, and the tmp_*.db will be saved in the current directory
model0_infdata = model0.sample(500, chains = 4, return_infdata = True)
model0_infdata
```

Then we used the arviz to diagnose the model. 

We can obtain the Rhat, ess, and other diagnostic for each parameter.

### Code cell 13

```python
az.summary(model0_infdata)
```

We could also plot the traceplot and posterior distribution for each parameter to check the convergence of the model.

Note: different chains are plotted in different colors.

### Code cell 15

```python
fig = az.plot_trace(model0_infdata, var_names=['a', 'v', 't', 'z'])
```

### <a id='toc1_1_2_'></a>[HDDM with regressors](#toc0_)

Next, we will try a more complex model, such as hddm with regressors. 
- In this example, we assume that the decision boundary varies with the condition `conf` and set the `LC` condition as a reference, so that the `v` coefficient represents the difference between LC and HC. 
- We show how to save a model to an arbitrary directory using the `save_name` argument. 
- `sample_prior` allow to sample draws from prior distribution, and then used to calculate Bayes Factor (BF). 
- To enable model comparison, we have added the `loglike = True` argument to calculate the pointwise log likelihood of the model, which can be further used for calculating WAIC and LOOIC. 
- Additionally, we have added the `ppc = True` argument, which allows us to generate posterior predictive checks for the model."

### Code cell 18

```python
%time
model_reg = hddm.HDDMRegressor(data, "v ~ 1 + C(conf, Treatment('LC'))",include = ['v', 'a', 't', 'z'])
# note: setting save_name argument will delete the _temp*.db file
save_name = "model_fitted/hddmregressor_example"
model_reg_infdata = model_reg.sample(
    500, chains = 4, 
    return_infdata = True, save_name = save_name, 
    sample_prior = True, loglike = True, ppc = True)
```

### Code cell 19

```python
model_reg_infdata
```

We can see the InfData consist of five parts, including the prior, posterior, posterior_predictive, log_likelhood, and observed_data.

When you set the 'save_name' argument to a `str` of the desired file path, hddm will save the model to two files in that location: the model itself (e.g. `hddmregressor_example.hddm`) and the trace of the model (`hddmregressor_example.db`). 

Additionally, if you use the argument `return_infdata = True`, it convert the hddm model to ArViz InferenceData, the InferenceData will also be saved to the same path (`hddmregressor_example.nc`).

As a result, we can easily load the model and InferenceData using the following code:

### Code cell 22

```python
%time
save_name = "model_fitted/hddmregressor_example"
# loading the inference data is faster than loading the origin hddm class
# model_reg = hddm.load(save_name + ".hddm")
model_reg_infdata = az.from_netcdf(save_name + ".nc")
```

Then we can use the InferenceData to calculate PSIS-LOO.

### Code cell 24

```python
%time
hddm_reg_LOO = az.loo(model_reg_infdata, var_name="log_lik", pointwise=True)
hddm_reg_LOO
```

### <a id='toc1_1_3_'></a>[model comparison with LOO](#toc0_)

To determine if the task condition `conf` affects the drift rate, we can compare `model0` and `model_reg` with ArviZ LOO-based model comparison.

### Code cell 27

```python
%time
# To compare models by LOO, we should make sure 
# that the pointwise log-likelihood is calculated.
model0_infdata = model0.to_infdata(loglike=True)

compare_dict = {
    "model0(baseline)": model0_infdata,
    "model_reg(v ~ conf)": model_reg_infdata,
}
model_compare = az.compare(compare_dict, var_name="log_lik", round_to="none")
model_compare
```

We can see that the model with varing of drift rate get higher elpd value that indicate it is better than the baseline model.

Futhermore, we can plot posterior predictive checks to inspect the best model. ArviZ 1.1 uses the newer `plot_ppc_dist()` interface for DataTree-style InferenceData. More examples see [dockerHDDM_Workflow](./dockerHDDM_Workflow.ipynb)

### Code cell 30

```python
az.plot_ppc_dist(model_reg_infdata, var_names="rt", num_samples=50)
```

## <a id='toc1_2_'></a>[statistical inference](#toc0_)

Finally, statistical inferences are made from the parameter posterior distributions of the optimal model, i.e., the effect of conf on drift rate `v`. The cell below plots the posterior distribution, 95% HDI, a zero reference line, and a ROPE from -0.2 to 0.2. The Bayes factor is then computed with `az.bayes_factor()`.

### Code cell 33

```python
import matplotlib.pyplot as plt

param_name = "v_C(conf, Treatment('LC'))[T.HC]"
posterior_samples = model_reg_infdata.posterior[param_name].values.ravel()
hdi = az.hdi(posterior_samples, prob=0.95)

fig, ax = plt.subplots(figsize=(4, 4))
ax.hist(posterior_samples, bins=40, density=True, alpha=0.75)
ax.axvspan(-0.2, 0.2, color="red", alpha=0.15, label="ROPE [-0.2, 0.2]")
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(hdi[0], color="C1", linestyle=":", label="95% HDI")
ax.axvline(hdi[1], color="C1", linestyle=":")
ax.set_title(param_name)
ax.set_xlabel("posterior sample")
ax.set_ylabel("density")
ax.legend()
plt.tight_layout()
```

### Code cell 34

```python
param_name = "v_C(conf, Treatment('LC'))[T.HC]"
bf = az.bayes_factor(model_reg_infdata, var_names=[param_name], ref_vals=0)
bf
```

In conclusion, our contribution has three key benefits:
1. You now have the freedom to save the model to any desired path, even if it is not available in the original hddm.
2. Rather than generating separate db files for each chain (which could not be moved to other paths in the previous version), we now save all chains' db files in a single location of your choosing.
3. We provide a simple way to convert the hddm model to ArViz InferenceData, which allows you to calculate pointwise loglikelihood for model comparison (e.g. WAIC and LOOIC), and generate predictions for model checking (e.g. posterior predictive checks)."

Lastly, I hope you enjoy using this version as it will be beneficial to your modeling work.

If you have any questions, please contact me at [<EMAIL>](<EMAIL>) or commit an issue at [epool/kabuki](https://gitee.com/epool/kabuki/issues) ↗
