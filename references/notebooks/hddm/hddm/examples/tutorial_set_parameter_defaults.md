# tutorial set parameter defaults

> Converted from `hddm/hddm/examples/tutorial_set_parameter_defaults.ipynb`. Code is preserved; rich outputs are omitted.

# Tutorial on Parameter defaults

As of version `0.9.8`, `HDDM` doesn't expect that you always explicitly want to fit the `v`, `a` and `t` parameters. You are now allowed to fix any of these parameters to any default you like. In this tutorial we show how to fit any given subset of parameters of a model, while supplying (user picked) default values for the remaining parameters.

## Install (colab)

### Code cell 3

```python
# package to help train networks
# !pip install git+https://github.com/AlexanderFengler/LANfactory

# package containing simulators for ssms
# !pip install git+https://github.com/AlexanderFengler/ssm_simulators

# packages related to hddm
# !pip install cython
# !pip install pymc==2.3.8
# !pip install git+https://github.com/hddm-devs/kabuki
# !pip install git+https://github.com/hddm-devs/hddm
```

## Load Modules

### Code cell 5

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

## Example Models

### `HDDM()`
#### Simulate Data

### Code cell 8

```python
# from hddm.simulators.hddm_dataset_generators import simulator_h_c
from hddm.simulators.basic_simulator import simulator
from hddm.simulators.hddm_dataset_generators import hddm_preprocess

model = "ddm_hddm_base"

data = simulator(theta=[1.0, 1.0, 0.5, 0.5], model=model, n_samples=500)

data = hddm_preprocess(data)
```

#### Model and  Sample

Let's first fit all parameters.

### Code cell 10

```python
hddm_model = hddm.HDDM(
    data,
    include=["v", "a", "t", "z"],
    informative=False,
    is_group_model=False,
)
```

### Code cell 11

```python
hddm_model.sample(1000, burn=500)
```

### Code cell 12

```python
hddm_model.gen_stats()
```

Now we **fix `a` to it's default** as per the `HDDM`-supplied `model_config` dictionary. As shown below,
this sets `a = 2.` which corresponds to an overestimation. We expect that, having fixed `a` at such value, we will correspondingly overestimate `v` to compensate (however the fit will end up worse in general).

### Code cell 14

```python
hddm.model_config.model_config["ddm_hddm_base"]
```

### Code cell 15

```python
hddm_model_no_a = hddm.HDDM(
    data,
    include=["v", "t", "z"],
    informative=False,
    is_group_model=False,
)
```

### Code cell 16

```python
hddm_model_no_a.sample(1000, burn=500)
```

### Code cell 17

```python
hddm_model_no_a.gen_stats()
```

As predicted, `v` is now overestimated as well.

Let's now try to set `a` to a default of our liking. We will set it to the ground-truth and again not include it in the parameters to estimate. To do so, we supply our own `model_config` to the `HDDM()` class.

### Code cell 19

```python
from copy import deepcopy

# copy model_config dictionary so we can change it
my_model_config = deepcopy(hddm.model_config.model_config["ddm_hddm_base"])

# setting 'a' to 1.
my_model_config["params_default"][1] = 1.0

hddm_model_no_a_2 = hddm.HDDM(
    data,
    include=["v", "t", "z"],
    informative=False,
    is_group_model=False,
    model_config=my_model_config,
)
```

### Code cell 20

```python
hddm_model_no_a_2.sample(1000, burn=500)
```

### Code cell 21

```python
hddm_model_no_a_2.gen_stats()
```

As we see, in this case `v` is estimated appropriately again.

##### Let's compare DICs

### Code cell 24

```python
print("Standard: ", hddm_model.dic)
print("No a with HDDM default: ", hddm_model_no_a.dic)
print("No a with a set to ground truth: ", hddm_model_no_a_2.dic)
```

### HDDMnn()

Let's repeat this with another model via the `HDDMnn()` class.
We will pick the `HDDM`-supplied `angle` model.

#### Simulate Data

### Code cell 27

```python
model = "angle"
theta = [1.0, 1.5, 0.5, 0.5, 0.2]  # v, a, z, t, theta
data_angle = simulator(theta=theta, model="angle", n_samples=500)
data_angle = hddm_preprocess(data_angle, keep_negative_responses=True)
```

#### Model and Sample

### Code cell 29

```python
model_angle = hddm.HDDMnn(
    data_angle, model="angle", include=["v", "a", "t", "z", "theta"]
)
```

### Code cell 30

```python
model_angle.sample(1000, burn=500)
```

### Code cell 31

```python
model_angle.gen_stats()
```

Again we will now leave out one parameter (let's pick `theta` this time). As we can see from the printed `model_config` below, the default that will be chosen for this parameter is to set it to `0` in this case.

### Code cell 33

```python
hddm.model_config.model_config
```

### Code cell 34

```python
model_angle_no_theta = hddm.HDDMnn(
    data_angle, model="angle", include=["v", "a", "t", "z"]
)
```

### Code cell 35

```python
model_angle_no_theta.sample(1000, burn=500)
```

### Code cell 36

```python
model_angle_no_theta.gen_stats()
```

Again we observe how the parameter estimates are affected by the *wrong choice of `theta`. The model tries to compensate for the parallel bounds (no collapse), implied by the `theta` default, by decreasing `a` and slightly increasing `v`. Let's now try again, but this time we set `theta` fixed to the actual *ground truth*.

### Code cell 38

```python
# copy out the model_config dictionary for the angle model
my_model_config_angle = deepcopy(hddm.model_config.model_config["angle"])
# set theta default to the ground truth defined above
my_model_config_angle["params_default"][4] = 0.2

model_angle_no_theta_2 = hddm.HDDMnn(
    data_angle,
    model="angle",
    include=["v", "a", "t", "z"],
    model_config=my_model_config_angle,
)
```

### Code cell 39

```python
model_angle_no_theta_2.sample(1000, burn=500)
```

### Code cell 40

```python
model_angle_no_theta_2.gen_stats()
```

As we see, fixing `theta` to the actual ground truth, corrects the parameter estimates of the remaining parameters to be much more accurate again.

##### Let's compare DICs

### Code cell 43

```python
print("Standard: ", model_angle.dic)
print("theta set to model_config default: ", model_angle_no_theta.dic)
print("theta set to ground truth: ", model_angle_no_theta_2.dic)
```

We observe in this case, that fixing `theta` to `0` instead of `0.2`, didn't do too much damage as far as the DICs are concerned. Nevertheless, the *explicitly wrong* model performs worst as per this metric.

##### END

Hopefully this was helpful.
