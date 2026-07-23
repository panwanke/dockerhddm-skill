# dockerHDDM PPC rt quantiles

> Converted from `dockerHDDMTutorial/dockerHDDM_PPC_rt_quantiles.ipynb`. Code is preserved; rich outputs are omitted.

### Code cell 1

```python
import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import hddm
import kabuki
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import xarray as xr

print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
print("The current version of arviz is: ", az.__version__)
```

### Code cell 2

```python
data = hddm.load_csv(hddm.__path__[0] + "/examples/cavanagh_theta_nn.csv")
data = data[data['subj_idx'].isin([0,1,2,3,4])]
data.head()
```

### Code cell 3

```python
%time
model_reg = hddm.HDDMRegressor(data, "v ~ 1 + C(conf, Treatment('LC'))",include = ['v', 'a', 't', 'z'])
# note: setting save_name argument will delete the _temp*.db file
save_name = "test/hddmregressor_example"
model_reg_infdata = model_reg.sample(
    500, chains = 4, 
    return_infdata = True, save_name = save_name, 
    sample_prior = True, loglike = False, ppc = True)
```

### Code cell 4

```python
model_reg_infdata
```

### Code cell 5

```python
%time
save_name = "test/hddmregressor_example"
# loading the inference data is faster than loading the origin hddm class
# model_reg = hddm.load(save_name + ".hddm")
model_reg_infdata = az.from_netcdf(save_name + ".nc")
```

### Code cell 6

```python
def plot_rt_quantiles_corrected(infdata, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Plot quantile-quantile comparison between observed and posterior predictive reaction times (RTs).
    Handles positive (correct/upper boundary) and negative (incorrect/lower boundary) RTs separately.
    
    Parameters
    ----------
    infdata : arviz.InferenceData
        InferenceData object containing observed_data and posterior_predictive groups.
    quantiles : list of float, optional
        Quantile levels to compute and plot (default: [0.1, 0.3, 0.5, 0.7, 0.9]).
    """
    
    # 1. Extract observed data
    obs_rt_all = infdata.observed_data["rt"]
    
    # 2. Extract posterior predictive data (stack chain and draw dimensions)
    pp_rt_all = infdata.posterior_predictive["rt"].stack(sample=("chain", "draw"))

    # 3. Automatically detect response type based on RT sign:
    #    RT > 0 -> Response 1 (upper boundary/correct)
    #    RT < 0 -> Response 0 (lower boundary/incorrect)
    #    Note: Using absolute values for quantile computation
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    
    # === Define two conditions: positive RTs and negative RTs ===
    conditions = [
        {"name": "Positive RTs (Upper Boundary)", "sign": 1},
        {"name": "Negative RTs (Lower Boundary)", "sign": -1}
    ]
    
    for i, cond in enumerate(conditions):
        ax = axes[i]
        sign = cond["sign"]
        
        # --- A. Process observed data ---
        # Filter: keep only data with the current sign
        if sign == 1:
            curr_obs = obs_rt_all.where(obs_rt_all > 0, drop=True)
        else:
            curr_obs = obs_rt_all.where(obs_rt_all < 0, drop=True)
            
        # Skip if no data (e.g., only positive responses)
        if curr_obs.size == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', fontsize=12)
            ax.set_title(cond["name"])
            continue

        # Take absolute values and compute quantiles
        # [Correction] Using dim="obs_id" for observed data
        curr_obs_abs = np.abs(curr_obs)
        obs_qs = curr_obs_abs.quantile(quantiles, dim="obs_id")
        
        # --- B. Process posterior predictive data ---
        # xarray's where operation works similarly for posterior data
        if sign == 1:
            curr_pp = pp_rt_all.where(pp_rt_all > 0, drop=True)
        else:
            curr_pp = pp_rt_all.where(pp_rt_all < 0, drop=True)
            
        # Take absolute values
        curr_pp_abs = np.abs(curr_pp)
        
        # Compute quantiles
        # Note: Posterior data typically retains the 'sample' dimension
        #       and aggregates over obs_id (trial) dimension
        # [Correction] Identify the non-sample dimension name dynamically
        pp_dim_name = [d for d in curr_pp_abs.dims if d not in ["sample", "chain", "draw"]][0]
        pp_qs = curr_pp_abs.quantile(quantiles, dim=pp_dim_name)
        
        # Calculate HDI (uncertainty intervals)
        # pp_qs has shape: (n_quantiles, n_samples)
        # Compute statistics across the sample dimension (axis=1)
        pp_qs_np = pp_qs.values
        
        # Compute mean and 94% interval for each quantile
        pp_mean = np.nanmean(pp_qs_np, axis=1)
        pp_lower = np.nanpercentile(pp_qs_np, 3, axis=1)
        pp_upper = np.nanpercentile(pp_qs_np, 97, axis=1)
        
        # --- C. Plotting ---
        # Plot posterior predictive interval (fan)
        ax.fill_between(
            quantiles, pp_lower, pp_upper, 
            color='C1', alpha=0.5, label='94% Posterior PPI'
        )
        ax.plot(
            quantiles, pp_mean, 'o-', 
            color='C1', label='Posterior Mean'
        )
        
        # Plot observed data
        ax.plot(
            quantiles, obs_qs, 'x--', 
            color='k', markersize=10, label='Observed Data'
        )
        
        ax.set_title(cond["name"])
        ax.set_xlabel("Quantile")
        ax.set_xticks(quantiles)
        if i == 0: 
            ax.set_ylabel("Reaction Time (|s|)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

### Code cell 7

```python
plot_rt_quantiles_corrected(model_reg_infdata)
```
