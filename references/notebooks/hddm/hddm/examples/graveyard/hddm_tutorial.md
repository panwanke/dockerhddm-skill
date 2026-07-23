# hddm tutorial

> Converted from `hddm/hddm/examples/graveyard/hddm_tutorial.ipynb`. Code is preserved; rich outputs are omitted.

# HDDM tutorial.

```text
Import the modules we are going to use. Pandas for the dataframe and matplotlib for plotting.
```

### Code cell 3

```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Code cell 4

```python
import hddm

print(hddm.__version__)

import kabuki

print(kabuki.__version__)
```

## Loading data

### Code cell 6

```python
!head PD_PS.csv
```

### Code cell 7

```python
data = pd.DataFrame(hddm.load_csv("PD_PS.csv"))
del data["conf"]
data = hddm.utils.flip_errors(data)
```

### Code cell 8

```python
data.head(10)
```

```text
To make things a little quicker and interactive, lets discard some of the subjects.
```

### Code cell 10

```python
subj_idx = data.subj_idx.unique()
data = data[data.subj_idx.isin(subj_idx[:8])]
data.subj_idx.unique()
```

```text
Lets look at the RT distributions
```

### Code cell 12

```python
fig = plt.figure()
ax = fig.add_subplot(111, xlabel="RT", ylabel="count", title="RT distributions")
for i, subj_data in data.groupby("subj_idx"):
    ax.hist(subj_data.rt, bins=20, histtype="step")
```

## The Drift-Diffusion model

![](http://ski.clps.brown.edu/hddm_docs/_images/DDM_drifts_w_labels.svg)

```text
Lets create a simple, hierarchical model.
```

### Code cell 16

```python
m = hddm.HDDM(data)
```

Here is what the graphical model representation looks like:

![](http://ski.clps.brown.edu/hddm_docs/_images/hier_model.svg)
$$\mu_{a} \sim \mathcal{N}(0, 1)$$
$$\mu_{z} \sim \mathcal{N}(0, 1)$$
$$\mu_{v} \sim \mathcal{N}(0, 1)$$
$$\mu_{ter} \sim \mathcal{N}(0, 1)$$
$$\mu_{sv} \sim \mathcal{N}(0, 1)$$
$$\mu_{sz} \sim \mathcal{N}(0, 1)$$
$$\mu_{ster} \sim \mathcal{N}(0, 1)$$

$$\sigma_{a} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{z} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{v} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{ter} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{sv} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{sz} \sim \mathcal{U}(1e^{-10}, 100)$$
$$\sigma_{ster} \sim \mathcal{U}(1e^{-10}, 100)$$

$$a_{i} \sim \mathcal{N}(\mu_{a}, \sigma_{a})$$
$$z_{i} \sim \mathcal{N}(\mu_{z}, \sigma_{z})$$
$$v_{i} \sim \mathcal{N}(\mu_{v}, \sigma_{v})$$
$$ter_{i} \sim \mathcal{N}(\mu_{ter}, \sigma_{ter})$$
$$sv_{i} \sim \mathcal{N}(\mu_{sv}, \sigma_{sv})$$
$$sz_{i} \sim \mathcal{N}(\mu_{sz}, \sigma_{sz})$$
$$ster_{i} \sim \mathcal{N}(\mu_{ster}, \sigma_{ster})$$

$$RT_{i, j} \sim wfpt(a_{i}, z_{i}, v_{i}, ter_{i}, sv_{i}, sz_{i}, ster_{i})$$

### Code cell 18

```python
# find a good starting point
m.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
m.sample(5000, burn=2000)
```

```text
Analyze model
```

### Code cell 20

```python
m.gen_stats()
```

### Code cell 21

```python
m.plot_posteriors(plot_subjs=True)
```

### Code cell 22

```python
m.plot_posterior_predictive(columns=3, figsize=(14, 10))
```

```text
Certainly there is more structure in our experiment, lets see if we can model LL, WL and WW conflicts.
```

### Code cell 24

```python
m_stim = hddm.HDDM(data, depends_on={"v": "stim"})
m_stim.find_starting_values()
m_stim.sample(5000, burn=3000)
```

### Code cell 25

```python
m_stim.gen_stats()
```

### Code cell 26

```python
m_stim.plot_posteriors()
```

### Code cell 27

```python
kabuki.analyze.plot_posterior_nodes(m_stim.nodes_db.node[["v(WW)", "v(LL)", "v(WL)"]])
```

```text
Seems like WW and LL are almost the same, lets see if we can combine them.
```

### Code cell 29

```python
# Add new column conf to data
data["conf"] = "LC"
data.conf[data.stim.isin(["LL", "WW"])] = "HC"
```

### Code cell 30

```python
m_conf = hddm.HDDM(data, depends_on={"v": "conf"})
m_conf.find_starting_values()
m_conf.sample(5000, burn=3000)
```

### Code cell 31

```python
m_conf.gen_stats()
```

### Code cell 32

```python
m_conf.plot_posteriors()
```

### Code cell 33

```python
kabuki.analyze.plot_posterior_nodes(m_conf.nodes_db.node[["v(HC)", "v(LC)"]])
```

```text
Lets compare the two models using DIC (lower is better).
```

### Code cell 35

```python
print "Lumped model DIC: %f" % m.dic_info()['DIC']
print "Stimulus model DIC: %f" % m_stim.dic_info()['DIC']
print "Conflict model DIC: %f" % m_conf.dic_info()['DIC']
```

## Inter-trial variabilities

### Code cell 37

```python
# generate new simulated data
data_sv, params = hddm.generate.gen_rand_data(
    params={"a": 2, "v": 1, "t": 0.3, "z": 0.3, "sv": 0.2, "st": 0, "sz": 0},
    size=200,
    subjs=8,
)
```

### Code cell 38

```python
m_inter = hddm.HDDM(data_sv, include=["v", "a", "t", "z", "sv"])
m_inter.find_starting_values()
m_inter.sample(5000, burn=3000)
```

### Code cell 39

```python
stats = m_inter.gen_stats()
stats[stats.index.isin(["sv", "z"])]
```

### Code cell 40

```python
m_inter.plot_posteriors(["sv", "sv_var", "z"])
```

```text
Terrible convergence for sv. It often works better when not even attempt to estimate individual subject parameters for sv and only do group mean.
```

### Code cell 42

```python
m_inter_group = hddm.HDDM(
    data_sv, include=["v", "a", "t", "z", "sv"], group_only_nodes=["sv"]
)
m_inter_group.find_starting_values()
m_inter_group.sample(5000, burn=3000)
```

### Code cell 43

```python
stats = m_inter_group.gen_stats()
stats[stats.index.isin(["sv", "z"])]
```

### Code cell 44

```python
m_inter_group.plot_posteriors(["sv", "z"])
```

## Running multiple chains (in parallel). This requires the git version of hddm, not 0.4.

### Code cell 46

```python
# define a stand-alone function that will be executed remotely
def run_model(id):
    import hddm
    import numpy

    numpy.random.seed(123)
    params = {"a": 2, "v": 1, "t": 0.3, "sv": 0, "st": 0, "sz": 0, "z": 0.5}
    data, _ = hddm.generate.gen_rand_data(params=params, size=300)
    m = hddm.HDDM(data)
    m.find_starting_values()
    m.sample(20000, burn=15000, dbname="db%i" % id, db="pickle")
    return m


from IPython.parallel import Client

job_queue = Client(profile="hddm")[:]
jobs = job_queue.map(run_model, range(8))
models = jobs.get()
```

### Code cell 47

```python
kabuki.analyze.gelman_rubin(models)
```

## Dealing with outliers

### Code cell 49

```python
outlier_data, params = hddm.generate.gen_rand_data(size=200, n_fast_outliers=10)
params
```

### Code cell 50

```python
m_no_outlier = hddm.HDDM(outlier_data)
m_no_outlier.sample(10000, burn=5000)
```

### Code cell 51

```python
m_no_outlier.plot_posterior_predictive(columns=1)
```

### Code cell 52

```python
m_no_outlier.plot_posterior_quantiles(columns=1)
```

### Code cell 53

```python
m_outlier = hddm.HDDM(outlier_data, p_outlier=0.05)
m_outlier.sample(10000, burn=5000)
```

### Code cell 54

```python
m_outlier.gen_stats()
```

### Code cell 55

```python
m_outlier.plot_posterior_predictive(columns=1)
```

### Code cell 56

```python
m_outlier.plot_posterior_quantiles(columns=1)
```

## Fitting regression models

### Code cell 58

```python
data = pd.DataFrame(hddm.load_csv("PD_PS.csv"))

# Define the function to estimate the regression
# lambda is a one-line function definition
reg_func = lambda args, cols: args[0] + args[1] * cols[:, 0]

# Define the regression descriptor where we specify
reg = {
    "func": reg_func,  # which function to use
    "args": ["a_intercept", "a_slope"],  # the input arguments (arbitrary)
    "covariates": "theta",  # the name of column to use for the covariate (will be the argument cols to the function above)
    "outcome": "a",  # which parameter to regress on
}
m_reg = hddm.HDDMRegressor(
    data,
    reg,
    depends_on={"a_slope": "dbs_inv", "v": "conf"},
    group_only_nodes=["a_intercept", "a_slope"],
)
```

### Code cell 59

```python
# Warning, takes a long time to execute!
# m_reg.sample(20000, burn=15000)
```

### Code cell 60

```python
kabuki.analyze.plot_posterior_nodes(
    m_reg.nodes_db.node[["a_slope(0)", "a_slope(1)"]], bins=15
)
```

For more details, see http://ski.clps.brown.edu/papers/Cavanagh_DBSEEG.pdf

## Posterior Predictive Checks

### Code cell 63

```python
from hddm.utils import post_pred_check

post_pred_check(m_no_outlier)
```

## IPython notebook foo

```text
Executing code in octave (or matlab)
```

### Code cell 66

```python
%load_ext octavemagic
```

### Code cell 67

```python
%%octave -s 500,500
b = [0.292893218813452, 0.585786437626905, 0.292893218813452];
a = [1,  0,  0.171572875253810];
freqz(b, a, 32);
```

```text
Execute code in R
```

### Code cell 69

```python
%load_ext rmagic
import numpy

X = np.array([0, 1, 2, 3, 4])
Y = np.array([3, 5, 4, 6, 7])
```

### Code cell 70

```python
%%R -i X,Y -o XYcoef
XYlm = lm(Y~X)
XYcoef = coef(XYlm)
print(summary(XYlm))
par(mfrow=c(2,2))
plot(XYlm)
```
