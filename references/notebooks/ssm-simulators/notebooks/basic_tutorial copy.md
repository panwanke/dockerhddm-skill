# basic tutorial copy

> Converted from `ssm-simulators/notebooks/basic_tutorial copy.ipynb`. Code is preserved; rich outputs are omitted.

### Quick Start

The `ssms` package serves two purposes. 

1. Easy access to *fast simulators of sequential sampling models*
   
2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

We provide two minimal examples here to illustrate how to use each of the two capabilities.

#### Install 

Let's start with *installing* the `ssms` package.

You can do so by typing,

`pip install git+https://github.com/AlexanderFengler/ssm_simulators`

in your terminal.

Below you find a basic tutorial on how to use the package.

#### Tutorial

### Code cell 5

```python
# Import necessary packages
import numpy as np
import pandas as pd
import ssms
```

#### Using the Simulators

Let's start with using the basic simulators. 
You access the main simulators through the  `ssms.basic_simulators.simulator` function.

To get an idea about the models included in `ssms`, use the `config` module.
The central dictionary with metadata about included models sits in `ssms.config.model_config`.

### Code cell 7

```python
# Check included models
list(ssms.config.model_config.keys())
```

### Code cell 8

```python
ssms.config.model_config['ddm_deadline']
```

### Code cell 9

```python
# Take an example config for a given model
ssms.config.model_config['ddm']
```

**Note:**
The usual structure of these models includes,

- Parameter names (`'params'`)
- Bounds on the parameters (`'param_bounds'`)
- A function that defines a boundary for the respective model (`'boundary'`)
- The number of parameters (`'n_params'`)
- Defaults for the parameters (`'default_params'`)
- The number of choices the process can produce (`'nchoices'`)

The `'hddm_include'` key concerns information useful for integration with the [hddm](https://github.com/hddm-devs/hddm) python package, which facilitates hierarchical bayesian inference for sequential sampling models. It is not important for the present tutorial.

### Code cell 11

```python
from ssms.basic_simulators import simulator
p_choice_vec = []
dline_tmp_vec = []
for dline_tmp in np.linspace(0.2, 5, 20):
    sim_out = simulator(model = 'ddm_deadline', 
                        theta = [1., 1., 0.5, 0.5, dline_tmp],
                        n_samples = 10000)
    p_choice_vec.append(np.sum(sim_out['choices'] == 1.) / sim_out['choices'].shape[0])
    dline_tmp_vec.append(dline_tmp)
```

### Code cell 12

```python
plt.plot(dline_tmp_vec, p_choice_vec)
```

### Code cell 13

```python
from matplotlib import pyplot as plt
plt.hist(sim_out['rts'][sim_out['rts'] != -999] * sim_out['choices'][sim_out['rts'] != -999], bins = 50)
```

### Code cell 14

```python
np.hstack([sim_out['rts'], sim_out['choices']])
```

The output of the simulator is a `dictionary` with three elements.

1. `rts` (array)
2. `choices` (array)
3. `metadata` (dictionary)

The `metadata` includes the named parameters, simulator settings, and more.

#### Using the Training Data Generators

The training data generators sit on top of the simulator function to turn raw simulations into usable training data for training machine learning algorithms aimed at posterior or likelihood armortization.

We will use the `data_generator` class from `ssms.dataset_generators`. Initializing the `data_generator` boils down to supplying two configuration dictionaries.

1. The `generator_config`, concerns choices as to what kind of training data one wants to generate.
2. The `model_config` concerns choices with respect to the underlying generative *sequential sampling model*. 

We will consider a basic example here, concerning data generation to prepare for training [LANs](https://elifesciences.org/articles/65074).

Let's start by peeking at an example `generator_config`.

### Code cell 17

```python
ssms.config.data_generator_config['lan']['mlp']
```

You usually have to make just few changes to this basic configuration dictionary.
An example below.

### Code cell 19

```python
from copy import deepcopy
# Initialize the generator config (for MLP LANs)
generator_config = deepcopy(ssms.config.data_generator_config['snpe'])
# Specify generative model (one from the list of included models mentioned above)
generator_config['dgp_list'] = 'angle' 
# Specify number of parameter sets to simulate
generator_config['n_parameter_sets'] = 100 
# Specify how many samples a simulation run should entail
generator_config['n_samples'] = 1000
```

Now let's define our corresponding `model_config`.

### Code cell 21

```python
model_config = ssms.config.model_config['angle']
print(model_config)
```

We are now ready to initialize a `data_generator`, after which we can generate training data using the `generate_data_training_uniform` function, which will use the hypercube defined by our parameter bounds from the `model_config` to uniformly generate parameter sets and corresponding simulated datasets.

### Code cell 23

```python
my_dataset_generator = ssms.dataset_generators.data_generator_snpe(generator_config = generator_config,
                                                                   model_config = model_config)
```

### Code cell 24

```python
training_data = my_dataset_generator.generate_data_training_uniform(save = True)
```

### Code cell 25

```python
new_features = {i: {'data': training_data[0][i]['features'], 'labels': training_data[0][i]['labels']} for i in range(len(training_data[0]))}
```

### Code cell 26

```python
training_data.keys()
```

### Code cell 27

```python
trainin
```

### Code cell 28

```python
training_data[0]
```

### Code cell 30

```python
max_n_trials = 3000
mydict = {0: {'features': np.zeros((max_n_trials, 2)), 'labels': np.ones(4)},
          1: {'features': np.zeros((max_n_trials, 2)), 'labels': np.ones(4)}}


n_trials = int(np.random.uniform(low = 500, high = 3000))
n_batch = 2

# Inside the dataloader
my_batch = np.zeros((n_batch, n_trials, 2))

for i in range(n_batch):
    my_batch[i, :, :] = mydict[i]['features'][np.random.choice(max_n_trials, n_trials, replace = False), :]
```

### Code cell 31

```python
my_batch.shape
```

### Code cell 32

```python
np.random.choice(10, 2, replace=False)
```

`training_data` is a dictionary containing four keys:

1. `data` the features for [LANs](https://elifesciences.org/articles/65074), containing vectors of *model parameters*, as well as *rts* and *choices*.
2. `labels` which contain approximate likelihood values
3. `generator_config`, as defined above
4. `model_config`, as defined above

You can now use this training data for your purposes. If you want to train [LANs](https://elifesciences.org/articles/65074) yourself, you might find the [LANfactory](https://github.com/AlexanderFengler/LANfactory) package helpful.

You may also simply find the basic simulators provided with the **ssms** package useful, without any desire to use the outputs into training data for amortization purposes.

##### END
