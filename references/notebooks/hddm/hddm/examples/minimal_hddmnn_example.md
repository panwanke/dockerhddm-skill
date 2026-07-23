# minimal hddmnn example

> Converted from `hddm/hddm/examples/minimal_hddmnn_example.ipynb`. Code is preserved; rich outputs are omitted.

## COLAB PREP
(Commented out for safety)

**Note**:
Changing colab runtime to gpu will benefit performance.

### Code cell 2

```python
# !pip install cython
# !pip install pymc==2.3.8 # backend probabilistic programming framework (DO NOT USE CONDA HERE)
# !pip install git+https://github.com/AlexanderFengler/ssms@main  # package for data simulation
# !pip install git+https://github.com/hddm-devs/kabuki # backbone package for hddm to connect to pymc
# !pip install git+https://github.com/hddm-devs/hddm

# # Optional
# !pip install torch torchvision torchaudio # The LAN extension makes use of these
```

## Module imports

### Code cell 4

```python
import pickle
from matplotlib import pyplot as plt

import hddm
import ssms
```

## Data Simulation

### Code cell 6

```python
from hddm.simulators import simulator

help(simulator)
```

Simulate from `angle` model! (Bound collaps linearly according to an `angle-parameter` `theta`).

First, check `model_config` dictionary for general information about the model.

### Code cell 8

```python
hddm.model_config.model_config["angle"]
```

Below, please find some specifics about the `key`s in the `model_config` dictionary.

`params` --> string names of all model parameters (spcifies order in which associated values are defined)

`param_bounds` --> list of lists, first provides *lower bounds*, second provides *upper bounds* on parameters

`boundary` --> function that provides boundary values for vector inputs of `t` given the named parameter (it's applied symmetrically around 0 as upper and lower bounds)

**Note**:

Our Neural Nets are trained on the parameter bounds listed under `param_bounds`, so they roughly specify what we consider a reasonable parameter space.

### Code cell 10

```python
out = simulator(model="angle", theta=[1.0, 1.5, 0.5, 0.5, 0.3], n_samples=1000)
```

`out` is a three-tuple with entries:
1. `rts` (reaction times) (array)
2. `choices` (array)
3. `metadata` (dict)

### Code cell 12

```python
import numpy as np

plt.hist(
    np.squeeze(out[0]) * np.squeeze(out[1]), histtype="step", bins=40, color="black"
)
plt.xlabel("reaction times")
plt.ylabel("freq")
plt.show()
```

## Fit with HDDM

### Code cell 14

```python
import numpy as np
import pandas as pd

# Get data into format accepted by hddm package (below)
data = pd.DataFrame(
    np.stack([np.squeeze(out[0]), np.squeeze(out[1])]).T, columns=["rt", "response"]
)
data["subject"] = 0
```

### Code cell 15

```python
data
```

### Code cell 16

```python
# Instantiate model
# Prior is uniform over hypercube specified by `param_bounds`
# in the model_config above
hddm_model = hddm.HDDMnn(
    data,
    model="angle",
    include=[
        "v",
        "a",
        "z",
        "t",
        "theta",
    ],  # can use 'hddm_include' key in model_config here
)
```

### Code cell 17

```python
# Sample (MCMC via slice sampler)
hddm_model.sample(1000, burn=500)
```

### Code cell 18

```python
hddm_model.gen_stats()
```

**Note**:

Depending on the model and ground truth parameters, mixing can be rough (the number of samples chose above is just for testing, result quality benefits from ramping up the number of MCMC samples).

### Code cell 20

```python
hddm_model.plot_posteriors()
plt.show()
```

## Access trained likelihood (networks)

You can get the predictions from the `likelihood aproximators` directly via the `network_inspectors` module.

Just in case it's useful.

The networks expects as inputs a `numpy.array` of `n_model_parameters + 2` dimensions. 
The first `n` parameters are (column wise) provided in the order found in under the `params` key in the `model_config` dictionary above. Finally a `rt` colum and a `choice` column.

### Code cell 22

```python
# Get the model
lan_angle = hddm.network_inspectors.get_torch_mlp(model="angle")
```

### Code cell 23

```python
# Make some random parameter set
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model="angle", param_dict=None, n_parameter_vectors=1
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

# Note: The networks expects float32 inputs
network_input = network_input.astype(np.float32)

# Show example output
print("Some network outputs")
print(lan_angle(network_input)[:10])  # printing the first 10 outputs
print("Shape")
print(lan_angle(network_input).shape)  # original shape of output
```

### Code cell 24

```python
network_input.shape
```

## P.S.:

There are many other models included in `hddm`.
Apart from the `angle` model, there is also the `weibull` model which uses a *scaled weibull cdf* as a bound and may be of more immediate interest. 

In general parameter recovery can be tricky, especially for the simple i.i.d. datasets.

### Code cell 26

```python
hddm.model_config.model_config.keys()
```

### Code cell 27

```python
hddm.model_config.model_config["weibull"]
```

## END
