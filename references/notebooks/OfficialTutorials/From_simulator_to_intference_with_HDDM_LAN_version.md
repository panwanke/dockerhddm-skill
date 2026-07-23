# From simulator to intference with HDDM LAN version

> Converted from `OfficialTutorials/From_simulator_to_intference_with_HDDM_LAN_version.ipynb`. Code is preserved; rich outputs are omitted.

# From simulator to inference with HDDM (LAN version)

### Code cell 2

```python
# package to help train networks
# %pip install git+https://github.com/AlexanderFengler/LANfactory
```

### Code cell 3

```python
# %conda install --quiet --yes scipy
```

### Code cell 4

```python
# Package to help train networks (explained above)
# import lanfactory

# Package containing simulators for ssms (explained above)
import ssms

# Other misc packages
import os
import numpy as np
from copy import deepcopy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch

import hddm
import kabuki
print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
```

### Code cell 5

```python
# MAKE CONFIGS
from ssms.config import data_generator_config

# Initialize the generator config (for MLP LANs)

# (We start from a supplied example in the ssms package)
# generator_config = deepcopy(data_generator_config['lan']['mlp'])
generator_config = deepcopy(data_generator_config['lan'])

# Specify generative model (one from the list of included models in the ssms package)
generator_config['dgp_list'] = 'ddm'

# Specify number of parameter sets to simulate
generator_config['n_parameter_sets'] = 5000

# Specify how many samples a simulation run should entail
generator_config['n_samples'] = 2000

# Specify how many training examples to extract from
# a single parametervector
generator_config['n_training_examples_by_parameter_set'] = 2000

# Specify folder in which to save generated data
generator_config['output_folder'] = 'lan_to_hddm_tmp_data/lan_mlp/'

# Make model config dict
model_config = ssms.config.model_config['ddm']

# Show
model_config
```

### Code cell 6

```python
generator_config
```

### Code cell 7

```python
my_dataset_generator = ssms.dataset_generators.data_generator(generator_config = generator_config,
                                                              model_config = model_config)

training_data = my_dataset_generator.generate_data_training_uniform(save = True)
```

### Code cell 8

```python
import pickle
import glob
```

### Code cell 9

```python
# load the file from the path in the above "writing to file"
fn = glob.glob('lan_to_hddm_tmp_data/lan_mlp/training_data_*.pickle')[0]
tmp_data = pickle.load(open(fn, "rb"))
```

Structure of training data:
`data`: Simulated data. Last column is choice and second to last it RT. Columns before are parameters that generated the observation. Has shape (`n_parameter_sets` x `n_training_samples_by_parameter_set`, `n_params` + 2)
`labels`: KDE of likelihood of simulated data. Used (?) as the labels to train the network. Has shape (`n_parameter_sets` x `n_training_samples_by_parameter_set`,)
`choice_p`: Thought these would be choice proportions for each parameter combination but they don't seem to match the proportions in simulated data.
`thetas`: Parameter combinations used to generate simulated data. Has shape (`n_parameter_sets` x `n_params`)
`binned_128`:
`binned_256`:
`generator_config`: Same as defined above
`model_config`: Same as defined above

### Code cell 11

```python
tmp_data.keys()
```

### Code cell 12

```python
tmp_data['data'].shape
```

### Code cell 13

```python
tmp_data['data']
```

### Code cell 14

```python
tmp_data['thetas']
```

### Code cell 15

```python
tmp_data['thetas'].shape
```

### Code cell 16

```python
# kde of likelihood for simulated data OR max negative RT
# called "label" because (?) used as labels for training the network    
tmp_data['labels']
```

### Code cell 17

```python
tmp_data['labels'].shape
```

### Code cell 18

```python
tmp_data['choice_p']
```

### Code cell 19

```python
tmp_data['choice_p'].shape
```

### Code cell 20

```python
[i[5] for i in tmp_data['data'][0:1000,]].count(1)/1000
```

### Code cell 21

```python
[i[5] for i in tmp_data['data'][(5000000-(999*2)+1):(5000000-999),]].count(1)/1000
```
