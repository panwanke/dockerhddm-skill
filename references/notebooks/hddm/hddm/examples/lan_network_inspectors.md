# lan network inspectors

> Converted from `hddm/hddm/examples/lan_network_inspectors.ipynb`. Code is preserved; rich outputs are omitted.

## Network Inspectors

The ```network_inspectors()``` module allows you to inspect the LANs directly. We will be grateful if you report any strange behavior you might find.

### Code cell 3

```python
# MODULE IMPORTS ----
import numpy as np
import hddm
```

### Direct access to batch predictions

You can use the ```hddm.network_inspectors.get_torch_mlp()``` function to access network predictions.

### Code cell 6

```python
# Specify model
model = "angle"
lan_angle = hddm.network_inspectors.get_torch_mlp(model=model)
```

### Code cell 7

```python
# Make some random parameter set
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model=model, param_dict=None, n_parameter_vectors=1
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

### Plotting Utilities

HDDM provides two plotting function to investigate the network outputs directly. The ```kde_vs_lan_likelihoods()``` plot and the ```lan_manifold()``` plot. 

**NOTE**:
These utilities are designed for 2-choice models at the moment.

#### `kde_vs_lan_likelihoods()`

### Code cell 11

```python
# Make some parameters
parameter_df = hddm.simulators.make_parameter_vectors_nn(
    model=model, param_dict=None, n_parameter_vectors=10
)
```

### Code cell 12

```python
parameter_df
```

### Code cell 13

```python
hddm.network_inspectors.kde_vs_lan_likelihoods(
    parameter_df=parameter_df, model=model, cols=3, n_samples=2000, n_reps=10, show=True
)
```

#### `lan_manifold()`

Lastly, you can use the ```lan_manifold()``` plot to investigate the LAN likelihoods over a range of parameters. 

The idea is to use a base parameter vector and vary one of the parameters in a prespecificed range. 

This plot can be informative if you would like to understand better how a parameter affects model behavior.

### Code cell 16

```python
# Now plotting
hddm.network_inspectors.lan_manifold(
    parameter_df=parameter_df,
    vary_dict={"v": np.linspace(-2, 2, 20)},
    model=model,
    n_rt_steps=300,
    fig_scale=1.0,
    max_rt=5,
    save=True,
    show=True,
)
```
