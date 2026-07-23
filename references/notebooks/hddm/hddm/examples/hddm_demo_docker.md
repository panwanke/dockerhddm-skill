# hddm demo docker

> Converted from `hddm/hddm/examples/hddm_demo_docker.ipynb`. Code is preserved; rich outputs are omitted.

# Demo

In this notebook, we reproduced the original demo within this HDDM docker container. Also, we demonstrated how to use `p_tqdm` for parallel processing.

This script reproduces the [official tutorial of HDDM](http://ski.clps.brown.edu/hddm_docs/tutorial_python.html) by the HDDM (0.9.0) in this docker image.

### Code cell 3

```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Code cell 4

```python
%matplotlib inline
import hddm
import kabuki

print(hddm.__version__)
```

## Loading data

### Code cell 6

```python
# find the directory of "cavanagh_theta_nn.csv"
print(hddm.__file__)
```

### Code cell 7

```python
!head /opt/conda/lib/python3.8/site-packages/hddm/examples/cavanagh_theta_nn.csv
```

### Code cell 8

```python
data = hddm.load_csv(
    "/opt/conda/lib/python3.8/site-packages/hddm/examples/cavanagh_theta_nn.csv"
)
data.head()
```

### Code cell 9

```python
data = hddm.utils.flip_errors(data)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="RT", ylabel="count", title="RT distributions")
for i, subj_data in data.groupby("subj_idx"):
    subj_data.rt.hist(bins=20, histtype="step", ax=ax)

# plt.savefig('hddm_demo_fig_00.pdf')
```

Check number of trials of each condition for each participant.

### Code cell 11

```python
data.groupby(["subj_idx", "stim"]).size()
```

## Fitting a hierarchical model

### Code cell 13

```python
# Instantiate model object passing it our data (no need to call flip_errors() before passing it).
# This will tailor an individual hierarchical DDM around your dataset.
m1 = hddm.HDDM(data)
# find a good starting point which helps with the convergence.
m1.find_starting_values()
# start drawing 7000 samples and discarding 5000 as burn-in
m1.sample(2000, burn=500)
```

### Code cell 14

```python
stats = m1.gen_stats()
stats[stats.index.isin(["a", "a_std", "a_subj.0", "a_subj.1"])]
```

### Code cell 15

```python
m1.plot_posteriors(["a", "t", "v", "a_std"])
```

## Calculate Gelman_rubin r hat.

### Option 1: using`for` loop

As we can see, it takes 20min 32s on a machine with `Intel® Core™ i7-10700 CPU @ 2.90GHz × 16`

### Code cell 17

```python
samples = 5000  # Cavanagh used 30,000 and 10, 000 burn.
burn = 1000
thin = 1
chains = 5
```

### Code cell 18

```python
%%time
models_1_ser = []
for i in range(chains):
    m = hddm.HDDM(data)
    m.find_starting_values()
    m.sample(samples, burn=burn)
    models_1_ser.append(m)
```

### Code cell 19

```python
hddm.analyze.gelman_rubin(models_1_ser)
```

### Option 2: using `p_tqdm` for parall processing
Here, we replace the `for` loop with parallel processing. As we can see here, the parall processing takes 4min 1s minutes on the same machine.

### Code cell 21

```python
# parallel processing related packages
from p_tqdm import p_map
from functools import partial
```

### Code cell 22

```python
def ms1(id, data=None, samples=None, burn=None, save_name="cavanagh2011_m1"):
    print("running model%i" % id)

    import hddm

    dbname = save_name + "_chain_%i.db" % id
    mname = save_name + "_chain_%i" % id
    m = hddm.HDDM(data)
    m.find_starting_values()
    m.sample(
        samples, burn=burn, dbname=dbname, db="pickle"
    )  # it's neccessary to save the model data
    m.save(mname)

    return m
```

### Code cell 23

```python
%%time
models_1_par = p_map(
    partial(ms1, data=data, samples=samples, burn=burn), range(chains)
)  # progess bar is a mess ;(
```

### Code cell 24

```python
hddm.analyze.gelman_rubin(models_1_par)
```

### Code cell 25

```python
models_1_par[0].plot_posterior_predictive(figsize=(14, 10))
```

### Code cell 26

```python
m_comb = kabuki.utils.concat_models(models_1_par)  # combine four chains together
```

### Code cell 27

```python
m_comb.plot_posterior_predictive(figsize=(14, 10))
```

### Model 2: `v` depends on stimulus

The original tutorial used code like this:

```
m_stim = hddm.HDDM(data, depends_on={'v': 'stim'})
m_stim.find_starting_values()
m_stim.sample(10000, burn=1000)
```

Here we defined a function for stimulus coding and used parallel processing

### Code cell 29

```python
def ms2(id, data=None, samples=None, burn=None, save_name="cavanagh2011_m2"):
    print("running model%i" % id)

    import hddm

    dbname = save_name + "_chain_%i.db" % id
    mname = save_name + "_chain_%i" % id

    m = hddm.HDDM(data, depends_on={"v": "stim"})
    m.find_starting_values()
    m.sample(samples, burn=burn, dbname=dbname, db="pickle")  # save the model data
    m.save(mname)

    return m
```

### Code cell 30

```python
%%time
# note: the samples, burn, and chains can be changed too
models_2 = p_map(partial(ms2, data=data, samples=samples, burn=burn), range(chains))
```

### Code cell 31

```python
m_stim_all = kabuki.utils.concat_models(models_2)
```

### Code cell 32

```python
v_WW, v_LL, v_WL = m_stim_all.nodes_db.node[["v(WW)", "v(LL)", "v(WL)"]]
hddm.analyze.plot_posterior_nodes([v_WW, v_LL, v_WL])
plt.xlabel("drift-rate")
plt.ylabel("Posterior probability")
plt.title("Posterior of drift-rate group means")
# plt.savefig('hddm_demo_fig_06.pdf')
```

### Code cell 33

```python
print("P(WW > LL) = ", (v_WW.trace() > v_LL.trace()).mean())
print("P(LL > WL) = ", (v_LL.trace() > v_WL.trace()).mean())
```

### Code cell 34

```python
print("Lumped model DIC: %f" % m1.dic)
print("Stimulus model DIC: %f" % m_stim_all.dic)
```

## Within-subject effects

### Code cell 36

```python
from patsy import dmatrix

dmatrix("C(stim, Treatment('WL'))", data.head(10))
```

The original code in the tutorial was:

```
m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))")
m_within_subj.sample(5000, burn=200)
```

Here we used four chains for parallel processing

### Code cell 38

```python
def run_m_reg(id, data=None, samples=None, burn=None, save_name="cavanagh2011_reg"):
    import hddm

    dbname = save_name + "_chain_%i.db" % id
    mname = save_name + "_chain_%i" % id

    m = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))")
    m.find_starting_values()
    m.sample(
        samples, burn=burn, dbname=dbname, db="pickle"
    )  # it's neccessary to save the model data
    m.save(mname)

    return m
```

### Code cell 39

```python
%%time
m_reg_list = p_map(
    partial(run_m_reg, data=data, samples=5000, burn=1000), range(chains)
)
```

### Code cell 40

```python
m_reg_all = kabuki.utils.concat_models(m_reg_list)
```

### Code cell 41

```python
v_WL, v_LL, v_WW = m_reg_all.nodes_db.loc[
    [
        "v_Intercept",
        "v_C(stim, Treatment('WL'))[T.LL]",
        "v_C(stim, Treatment('WL'))[T.WW]",
    ],
    "node",
]
hddm.analyze.plot_posterior_nodes([v_WL, v_LL, v_WW])
plt.xlabel("drift-rate")
plt.ylabel("Posterior probability")
plt.title("Group mean posteriors of within-subject drift-rate effects.")
# plt.savefig('hddm_demo_fig_07.pdf')
```

## Fitting regression models

### Code cell 43

```python
%%time
m_reg = hddm.HDDMRegressor(
    data[data.dbs == 0], "a ~ theta:C(conf, Treatment('LC'))", depends_on={"v": "stim"}
)
m_reg.sample(5000, burn=1000)
```

### Code cell 44

```python
theta = m_reg.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
hddm.analyze.plot_posterior_nodes([theta], bins=20)
plt.xlabel("Theta coeffecient in ")
print("P(a_theta < 0) = ", (theta.trace() < 0).mean())
```

### Code cell 45

```python
%%time
m_reg_off = hddm.HDDMRegressor(
    data[data.dbs == 1], "a ~ theta:C(conf, Treatment('LC'))", depends_on={"v": "stim"}
)
m_reg_off.sample(5000, burn=1000)
```

### Code cell 46

```python
theta = m_reg_off.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
hddm.analyze.plot_posterior_nodes([theta], bins=10)
print("P(a_theta > 0) = ", (theta.trace() > 0).mean())
```

## Dealing with outliers

This part of the tutorial seems outdated because, since 0.6.0, HDDM has a default setting `p_outliers=.05`.

### Code cell 48

```python
outlier_data, params = hddm.generate.gen_rand_data(
    params={"a": 2, "t": 0.4, "v": 0.5}, size=200, n_fast_outliers=10
)
```

### Code cell 49

```python
m_no_outlier = hddm.HDDM(outlier_data)
m_no_outlier.sample(2000, burn=50)
```

### Code cell 50

```python
m_no_outlier.plot_posterior_predictive()
plt.title("Posterior predictive")
plt.xlabel("RT")
plt.ylabel("Probability density")
# plt.savefig('hddm_demo_fig_10.pdf')
```

### Code cell 51

```python
m_outlier = hddm.HDDM(outlier_data, p_outlier=0.05)
m_outlier.sample(2000, burn=20)
```

### Code cell 52

```python
m_outlier.plot_posterior_predictive()
plt.title("Posterior predictive")
plt.xlabel("RT")
plt.ylabel("Probability density")
# plt.savefig('hddm_demo_fig_11.pdf')
```
