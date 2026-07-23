# demo HDDMnnRL

> Converted from `hddm/hddm/examples/demo_HDDMnnRL/demo_HDDMnnRL.ipynb`. Code is preserved; rich outputs are omitted.

# Tutorial for analyzing instrumental learning data with the HDDMnnRL module

### Code cell 2

```python
import hddm
import pickle
import pandas as pd
```

### Code cell 3

```python
# with open('./rlssm_data.pickle', 'rb') as handle:
#     datafile = pickle.load(handle)
# print(len(datafile))

# dataset = datafile[0]
# data = hddm.utils.get_dataset_as_dataframe_rlssm(dataset)
# data.to_csv('rlssm_data.csv', index=False)
```

#### Load the data

### Code cell 5

```python
# Load the data
data = pd.read_csv("rlssm_data.csv")
```

#### Initialize the HDDMnnRL model and sample

### Code cell 7

```python
# Specify number of samples and burnins
nsamples = 100
nburn = 50
```

### Code cell 8

```python
m = hddm.HDDMnnRL(
    data,
    model="angle",
    rl_rule="RWupdate",
    non_centered=True,
    include=["v", "a", "t", "z", "theta", "rl_alpha"],
    p_outlier=0.0,
)
m.sample(nsamples, burn=nburn, dbname="traces.db", db="pickle")
```

#### Save the model

### Code cell 10

```python
# Save the model
m.save("rlssm_model")
```

### Code cell 11

```python
# Load the model
model = hddm.load("rlssm_model")
```

#### Check the posterior results

### Code cell 13

```python
m.plot_posteriors()
```

### Code cell 14

```python
# Load the trace
with open("./traces.db", "rb") as handle:
    tracefile = pickle.load(handle)
```

### Code cell 15

```python
# Re-format traces as a dataframe
traces = hddm.utils.get_traces_rlssm(tracefile)
```

### Code cell 16

```python
model_ssm = "angle"
model_rl = "RWupdate"

config_ssm = hddm.model_config.model_config[model_ssm]
config_rl = hddm.model_config_rl.model_config_rl[model_rl]
```

### Code cell 17

```python
_ = hddm.plotting.plot_posterior_pairs_rlssm(
    tracefile, config_ssm["params"] + config_rl["params"]
)
```

#### Posterior Predictive Checks

### Code cell 19

```python
num_posterior_samples = 3
p_lower = {0: 0.15, 1: 0.30, 2: 0.45}
p_upper = {0: 0.85, 1: 0.70, 2: 0.55}
ppc_sdata = hddm.plotting.gen_ppc_rlssm(
    model_ssm,
    config_ssm,
    model_rl,
    config_rl,
    data,
    traces,
    num_posterior_samples,
    p_lower,
    p_upper,
    save_data=True,
    save_name="ppc_data",
)
```

### Code cell 20

```python
# Load the saved ppc data
# ppc_sdata = pd.read_csv('./ppc_data.csv')
```

### Code cell 21

```python
_ = hddm.plotting.plot_ppc_choice_rlssm(data, ppc_sdata, 40, 10)
```

### Code cell 22

```python
_ = hddm.plotting.plot_ppc_rt_rlssm(data, ppc_sdata, 40, 0.06)
```
