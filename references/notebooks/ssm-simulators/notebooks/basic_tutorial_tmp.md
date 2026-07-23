# basic tutorial tmp

> Converted from `ssm-simulators/notebooks/basic_tutorial_tmp.ipynb`. Code is preserved; rich outputs are omitted.

### Quick Start

The `ssms` package serves two purposes. 

1. Easy access to *fast simulators of sequential sampling models*
   
2. Support infrastructure to construct training data for various approaches to likelihood / posterior amortization

We provide two minimal examples here to illustrate how to use each of the two capabilities.

Let's start with *installing* the `ssms` package.

You can do so by typing,

`pip install git+https://github.com/AlexanderFengler/ssm_simulators`

in your terminal.

### Code cell 3

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

### Code cell 5

```python
# Check included models
list(ssms.config.model_config.keys())[:12]
```

### Code cell 6

```python
# Take an example config for a given model
ssms.config.model_config['ds_conflict_drift']
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

### Code cell 8

```python
from ssms.basic_simulators import simulator

theta = ssms.config.model_config['ds_conflict_drift']['default_params'] 
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('tfixedp')] = 5
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('a')] = 1.5
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('tinit')] = 5
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('dinit')] = 0
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('tcoh')] = 1
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('dcoh')] = 0
theta[ssms.config.model_config['ds_conflict_drift']['params'].index('t')] = 0
sim_out = simulator(model = 'ds_conflict_drift', 
                    theta = theta,
                    n_samples = 1000)
```

### Code cell 9

```python
from ssms.basic_simulators import simulator

theta = ssms.config.model_config['ddm']['default_params'] 
theta[ssms.config.model_config['ddm']['params'].index('v')] = 5
theta[ssms.config.model_config['ddm']['params'].index('a')] = 1.5
theta[ssms.config.model_config['ddm']['params'].index('t')] = 0
sim_out_base = simulator(model = 'ddm', 
                    theta = theta,
                    n_samples = 1000)
```

### Code cell 10

```python
ssms.config.model_config['ds_conflict_drift']['params']
```

### Code cell 11

```python
# Plotting
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
plt.plot(sim_out['metadata']['drift'])
```

### Code cell 12

```python
plt.hist(sim_out['rts'] * sim_out['choices'], bins = 30, histtype = 'step')
plt.hist(sim_out_base['rts'] * sim_out_base['choices'], bins = 30, histtype = 'step')

plt.xlim((-3, 3))
```

The output of the simulator is a `dictionary` with three elements.

1. `rts` (array)
2. `choices` (array)
3. `metadata` (dictionary)

The `metadata` includes the named parameters, simulator settings, and more.

####

### Code cell 15

```python
# TEST STANDARD TRAINING DATA GENERATOR
# my_data_config = config.data_generator_config['lan']['cnn']
my_data_config = ssms.config.data_generator_config['lan']['mlp']
my_data_config['dgp_list'] = 'tradeoff_no_bias'
my_data_config['n_parameter_sets'] = 100
my_data_config['n_samples'] = 1000
```

### Code cell 16

```python
my_model_config = ssms.config.model_config['tradeoff_no_bias']
```

### Code cell 17

```python
my_dataset_generator = ssms.dataset_generators.data_generator(generator_config = my_data_config,
                                                              model_config = my_model_config)
```

### Code cell 18

```python
x = my_dataset_generator.generate_data_training_uniform(save = False)
```

### Code cell 19

```python
x
```

### Code cell 20

```python
# TEST NESTED MODEL TRAINING DATA
# my_data_config = config.data_generator_config['lan']['cnn']
my_data_config = ssms.config.data_generator_config['lan']['mlp']
my_data_config['dgp_list'] = 'glob'
my_data_config['n_parameter_sets'] = 100
my_data_config['n_samples'] = 1000
```

### Code cell 21

```python
my_model_config = ssms.config.model_config['glob']
```

### Code cell 22

```python
my_dataset_generator = ssms.dataset_generators.data_generator(generator_config = my_data_config,
                                                              model_config = my_model_config)
```

### Code cell 23

```python
my_output = my_dataset_generator.generate_data_nested(save = False)
```

### Code cell 24

```python
my_output
```

### Code cell 25

```python
# NEEDS NEW CONFIG HERE
my_output = my_dataset_generator.generate_data_ratio_estimator(save = False)
```

### Code cell 26

```python
my_output['labels']
```

### Code cell 27

```python
my_dataset_generator.model_config
```

### Code cell 28

```python
x['data'].shape
```

### Code cell 29

```python
x['labels'].shape
```

### Code cell 30

```python
my_output['label_parameters']
```

### Code cell 31

```python
from matplotlib import pyplot as plt
def normalized_gamma(t, alpha, tau, c):
    num_ = np.power(t, alpha - 1) * np.exp(np.divide(-t, tau))
    div_ = np.power(alpha - 1, alpha - 1) * np.power(tau, alpha - 1) * np.exp(- (alpha - 1))
    return c * np.divide(num_, div_)
```

### Code cell 32

```python
t = np.arange(0, 20, 0.01)
```

### Code cell 33

```python
for i in np.arange(1, 0.01, -0.01):
    out = ssms.basic_simulators.gamma_drift(t = t, shape = 10, scale = i, c = 2)
    plt.plot(t, out, alpha = i / 10, color = 'black')
```

### Code cell 34

```python
out = ssms.basic_simulators.gamma_drift(t = t, shape = 1.2, scale = 0.5, c = 2)
plt.plot(t, out, alpha = 5 / 10, color = 'black')
```

### Code cell 35

```python
out = normalized_gamma(t = t, alpha = 10, tau = 0.5, c = 2)
plt.plot(t, out, alpha = 1, color = 'black')
```

### Code cell 36

```python
plt.plot(t, out)
```

### Code cell 37

```python
np.zeros(100).shape
```

### Code cell 38

```python
from ssms.basic_simulators import simulator
model = 'gamma_drift_angle'
params_tmp = ssms.config.model_config[model]['default_params']
params_tmp[ssms.config.model_config[model]['params'].index('c')] = 3.0
params_tmp[ssms.config.model_config[model]['params'].index('theta')] = 0.0

sim_out = simulator(model = model, 
                    theta = params_tmp,
                    n_samples = 10000)
```

### Code cell 39

```python
from ssms.basic_simulators import simulator
model = 'angle'
params_tmp = ssms.config.model_config[model]['default_params']

sim_out = simulator(model = model, 
                    theta = params_tmp,
                    n_samples = 10000)
```

### Code cell 40

```python
sim_out['rts'].shape
```

### Code cell 41

```python
ssms.config.model_config['gamma_drift']['default_params']
```

### Code cell 42

```python
from matplotlib import pyplot as plt
plt.hist(sim_out['rts'] * sim_out['choices'], bins = 30)
```

### Code cell 43

```python
(sim_out['choices'] == 1).sum() / sim_out['choices'].shape[0]
```

# Test new tradeoff model

### Code cell 46

```python
from ssms.basic_simulators import simulator
model = 'tradeoff_angle_no_bias'
color_dict = {0: 'red',
              1: 'green',
              2: 'blue',
              3: 'black'}

rt_means_by_choice = {i: [] for i in range(4)}

params_tmp = ssms.config.model_config[model]['default_params']
params_tmp[ssms.config.model_config[model]['params'].index('d')] = 1.0
params_tmp[ssms.config.model_config[model]['params'].index('vh')] = 1.0
params_tmp[ssms.config.model_config[model]['params'].index('vl2')] = 1.0 #1.0
params_tmp[ssms.config.model_config[model]['params'].index('vl1')] = 0.0
params_tmp[ssms.config.model_config[model]['params'].index('theta')] = 0.0


#params_tmp[ssms.config.model_config[model]['params'].index('theta')] = 0.0
for k in range(10):
    sim_out = simulator(model = model, 
                        theta = params_tmp,
                        n_samples = 50000)
    for i in np.unique(sim_out['choices']):
        plt.hist(sim_out['rts'][sim_out['choices'] == i], histtype = 'step', bins = 30, color = color_dict[i], alpha = 1) #0.1)
        rt_means_by_choice[i].append(np.mean(sim_out['rts'][sim_out['choices'] == i]))
```

### Code cell 47

```python
from ssms.basic_simulators import simulator
model = 'tradeoff_angle_no_bias'
color_dict = {0: 'red',
              1: 'green',
              2: 'blue',
              3: 'black'}

rt_means_by_choice_2 = {i: [] for i in range(4)}
params_tmp = ssms.config.model_config[model]['default_params']
params_tmp[ssms.config.model_config[model]['params'].index('d')] = 0.0
params_tmp[ssms.config.model_config[model]['params'].index('vh')] = 1.0
params_tmp[ssms.config.model_config[model]['params'].index('vl2')] = 1.0 #1.0
params_tmp[ssms.config.model_config[model]['params'].index('vl1')] = 0.0
params_tmp[ssms.config.model_config[model]['params'].index('theta')] = 0.0

for k in range(10):
    sim_out = simulator(model = model, 
                        theta = params_tmp,
                        n_samples = 50000)
    for i in np.unique(sim_out['choices']):
        plt.hist(sim_out['rts'][sim_out['choices'] == i], histtype = 'step', bins = 30, color = color_dict[i], alpha = 1)
        rt_means_by_choice_2[i].append(np.mean(sim_out['rts'][sim_out['choices'] == i]))
```

### Code cell 48

```python
for i in range(4):
    plt.hist(rt_means_by_choice[i], histtype = 'step', color = 'red')
    plt.hist(rt_means_by_choice_2[i], histtype = 'step', color = 'black')
    plt.show()
```

### Code cell 49

```python
from matplotlib import pyplot as plt



for i in np.unique(sim_out['choices']):
    plt.hist(sim_out['rts'][sim_out['choices'] == i], histtype = 'step', bins = 30, color = color_dict[i], alpha = 0.5)
```

### Code cell 50

```python
np.tile(np.array([0], dtype = np.float32), 10).shape
```
