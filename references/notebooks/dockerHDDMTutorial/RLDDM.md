# RLDDM

> Converted from `dockerHDDMTutorial/RLDDM.ipynb`. Code is preserved; rich outputs are omitted.

### Code cell 1

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import hddm
import kabuki
import arviz as az

import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
print("The current version of arviz is: ", az.__version__)
```

## RLHDDM

More detials please see [RLHDDM](https://hddm.readthedocs.io/en/latest/demo_RLHDDMtutorial.html)

Few things to note:
- Due to the RLHDDM cannot be serialized by pickle, so
  - RLHDDM cannot be sampling parallalelly.
  - RLHDDM cannot be saved. 
  - But the infdata can be saved.

## Fitting

### load data

### Code cell 5

```python
data = hddm.load_csv(hddm.__path__[0] + "/examples/rlddm_data.csv")
data
```

### define and fit models

### Code cell 7

```python
nmcmc = 1500
n_burn = 500
n_chain = 4
```

### Code cell 8

```python
%time
# set dual=True to model separate learning rates for positive and negative prediction errors.
rl = hddm.HDDMrl(data, dual=True, include=["a","v","t","z","alpha"])
# set sample and burn-in
rl_infdata = rl.sample(nmcmc, burn=n_burn, chains=n_chain, return_infdata=True,save_name="rl", parallel=False)
```

### Code cell 9

```python
%time
# set dual=True to model separate learning rates for positive and negative prediction errors.
rl_dual = hddm.HDDMrl(data, dual=True, include=["a","v","t","z","alpha"])
# set sample and burn-in
rl_dual_infdata = rl_dual.sample(nmcmc, burn=n_burn, chains=n_chain, return_infdata=True,save_name="rl_dual", parallel=False)
```

### Code cell 10

```python
# rl_dual = hddm.HDDMrl(data, dual=True, include=["a","v","t","z","alpha"])
# rl_dual = rl_dual.load_db("rl_dual.db", db="pickle")
# rl_dual_infdata = az.from_netcdf("rl_dual.nc")
```

## Diagnostic

### Code cell 12

```python
az.summary(rl_dual_infdata, var_names = ['~subj', '~std'], filter_vars= 'regex')
```

### Code cell 13

```python
axes = az.plot_trace(rl_dual_infdata, var_names = ['~subj', '~std'], filter_vars= 'regex')
```

## Comparison

Use ArviZ 1.1 LOO-based model comparison. The pointwise log likelihood must be present in both DataTree objects.

### Code cell 15

```python
%time
rl_infdata = rl.to_infdata(loglike=True, save_name="rl", parallel=False)
rl_dual_infdata = rl_dual.to_infdata(loglike=True, save_name="rl_dual", parallel=False)
```

### Code cell 16

```python
compare_dict = {
    "Model 0 (RL)": rl_infdata,
    "Model 1 (RL with dual learning rate)": rl_dual_infdata,
}

az.compare(compare_dict, var_name="log_lik", round_to="none")
```

## PPC

### Code cell 18

```python
# create empty dataframe to store simulated data
sim_data = pd.DataFrame()
# create a column samp to be used to identify the simulated data sets
data["samp"] = 0
# get traces, note here we extract traces from rl_dual
traces = rl_dual.get_traces()
# decide how many times to repeat simulation process. repeating this multiple times is generally recommended as it better captures the uncertainty in the posterior distribution, but will also take some time
for i in tqdm(range(1, 51)):
    # randomly select a row in the traces to use for extracting parameter values
    sample = np.random.randint(0, traces.shape[0] - 1)
    # loop through all subjects in observed data
    for s in data.subj_idx.unique():
        # get number of trials for each condition.
        size0 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 0)].trial.unique()
        )
        size1 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 1)].trial.unique()
        )
        size2 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 2)].trial.unique()
        )
        # set parameter values for simulation
        a = traces.loc[sample, "a_subj." + str(s)]
        t = traces.loc[sample, "t_subj." + str(s)]
        scaler = traces.loc[sample, "v_subj." + str(s)]
        # when generating data with two learning rates pos_alpha represents learning rate for positive prediction errors and alpha for negative prediction errors
        alphaInv = traces.loc[sample, "alpha_subj." + str(s)]
        pos_alphaInv = traces.loc[sample, "pos_alpha_subj." + str(s)]
        # NOTE: take inverse logit of estimated alpha and pos_alpha
        alpha = np.exp(alphaInv) / (1 + np.exp(alphaInv))
        pos_alpha = np.exp(pos_alphaInv) / (1 + np.exp(pos_alphaInv))
        # simulate data for each condition changing only values of size, p_upper, p_lower and split_by between conditions.
        sim_data0 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size0,
            p_upper=0.8,
            p_lower=0.2,
            split_by=0,
        )
        sim_data1 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size1,
            p_upper=0.7,
            p_lower=0.3,
            split_by=1,
        )
        sim_data2 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size2,
            p_upper=0.6,
            p_lower=0.4,
            split_by=2,
        )
        # append the conditions
        # sim_data0 = sim_data0.append([sim_data1, sim_data2], ignore_index=True)
        sim_data0 = pd.concat([sim_data0, sim_data1, sim_data2], axis=0, ignore_index=True)
        # assign subj_idx
        sim_data0["subj_idx"] = s
        # identify that these are simulated data
        sim_data0["type"] = "simulated"
        # identify the simulated data
        sim_data0["samp"] = i
        # append data from each subject
        # sim_data = sim_data.append(sim_data0, ignore_index=True)
        sim_data = pd.concat([sim_data, sim_data0], axis=0, ignore_index=True)
# combine observed and simulated data
ppc_dual_data = data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "samp"]
].copy()
ppc_dual_data["type"] = "observed"
ppc_dual_sdata = sim_data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "type", "samp"]
].copy()
# ppc_dual_data = ppc_dual_data.append(ppc_dual_sdata)
ppc_dual_data = pd.concat([ppc_dual_data, ppc_dual_sdata], axis=0, ignore_index=True)
```

### Code cell 19

```python
import pymc
plot_ppc_dual_data = ppc_dual_data[ppc_dual_data.trial < 41].copy()

# bin trials to for smoother estimate of response proportion across learning
plot_ppc_dual_data["bin_trial"] = pd.cut(
    plot_ppc_dual_data.trial, 11, labels=np.linspace(0, 10, 11)
).astype("int64")
# calculate means for each sample
sums = (
    plot_ppc_dual_data.groupby(["bin_trial", "split_by", "samp", "type"])
    .mean()
    .reset_index()
)
# calculate the overall mean response across samples
ppc_dual_sim = sums.groupby(["bin_trial", "split_by", "type"]).mean().reset_index()
# initiate columns that will have the upper and lower bound of the hpd
ppc_dual_sim["upper_hpd"] = 0
ppc_dual_sim["lower_hpd"] = 0
for i in range(0, ppc_dual_sim.shape[0]):
    # calculate the hpd/hdi of the predicted mean responses across bin_trials
    hdi = pymc.utils.hpd(
        sums.response[
            (sums["bin_trial"] == ppc_dual_sim.bin_trial[i])
            & (sums["split_by"] == ppc_dual_sim.split_by[i])
            & (sums["type"] == ppc_dual_sim.type[i])
        ],
        alpha=0.1,
    )
    ppc_dual_sim.loc[i, "upper_hpd"] = hdi[1]
    ppc_dual_sim.loc[i, "lower_hpd"] = hdi[0]
# calculate error term as the distance from upper bound to mean
ppc_dual_sim["up_err"] = ppc_dual_sim["upper_hpd"] - ppc_dual_sim["response"]
ppc_dual_sim["low_err"] = ppc_dual_sim["response"] - ppc_dual_sim["lower_hpd"]
ppc_dual_sim["model"] = "RLDDM_dual_learning"
```

### Code cell 20

```python
# bin trials to for smoother estimate of response proportion across learning
plot_ppc_dual_data["bin_trial"] = pd.cut(
    plot_ppc_dual_data.trial, 11, labels=np.linspace(0, 10, 11)
).astype("int64")
# calculate means for each sample
sums = (
    plot_ppc_dual_data.groupby(["bin_trial", "split_by", "samp", "type"])
    .mean()
    .reset_index()
)
# calculate the overall mean response across samples
ppc_dual_sim = sums.groupby(["bin_trial", "split_by", "type"]).mean().reset_index()
# initiate columns that will have the upper and lower bound of the hpd
ppc_dual_sim["upper_hpd"] = 0
ppc_dual_sim["lower_hpd"] = 0
for i in range(0, ppc_dual_sim.shape[0]):
    # calculate the hpd/hdi of the predicted mean responses across bin_trials
    hdi = pymc.utils.hpd(
        sums.response[
            (sums["bin_trial"] == ppc_dual_sim.bin_trial[i])
            & (sums["split_by"] == ppc_dual_sim.split_by[i])
            & (sums["type"] == ppc_dual_sim.type[i])
        ],
        alpha=0.1,
    )
    ppc_dual_sim.loc[i, "upper_hpd"] = hdi[1]
    ppc_dual_sim.loc[i, "lower_hpd"] = hdi[0]
# calculate error term as the distance from upper bound to mean
ppc_dual_sim["up_err"] = ppc_dual_sim["upper_hpd"] - ppc_dual_sim["response"]
ppc_dual_sim["low_err"] = ppc_dual_sim["response"] - ppc_dual_sim["lower_hpd"]
ppc_dual_sim["model"] = "RLDDM_dual_learning"
```

### Code cell 21

```python
# plotting evolution of choice proportion for best option across learning for observed and simulated data.
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True)
for i in range(0, 3):
    ax = axs[i]
    d = ppc_dual_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "simulated")]
    ax.errorbar(
        d.bin_trial,
        d.response,
        yerr=[d.low_err, d.up_err],
        label="simulated",
        color="orange",
    )
    d = ppc_dual_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "observed")]
    ax.plot(d.bin_trial, d.response, linewidth=3, label="observed")
    ax.set_title("split_by = %i" % i, fontsize=20)
    ax.set_ylabel("mean response")
    ax.set_xlabel("trial")
plt.legend()
```
