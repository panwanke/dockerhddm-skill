# Posterior Predictive Checks

> Converted from `OfficialTutorials/Posterior_Predictive_Checks.ipynb`. Code is preserved; rich outputs are omitted.

### Code cell 1

```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import hddm
import kabuki
print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
```

### Code cell 2

```python
data, params = hddm.generate.gen_rand_data(params={'easy': {'v': 1, 'a': 2, 't': .3},
                                                   'hard': {'v': 1, 'a': 2, 't': .3}})
```

### Code cell 3

```python
m = hddm.HDDM(data, depends_on={'v': 'condition'},
             include = ['v', 'a', 't', 'z'],
                        informative = True,
                        is_group_model = False)
# save_name argument provided by our modified version of kabuki
m.sample(1000, burn=20, save_name = "test/model_for_ppc")
```

### Code cell 4

```python
print(m.nodes_db)
```

### Code cell 5

```python
hddm.analyze.plot_posterior_nodes(m.nodes_db.loc[['v(easy)', 'v(hard)'], 'node'])
```

### Code cell 6

```python
%%time
ppc_data = hddm.utils.post_pred_gen(m)
ppc_data
```

### Code cell 7

```python
# an alternative way generate posterior prediction by using our methods 
m.to_infdata(ppc=True)
```

### Code cell 8

```python
hddm.utils.post_pred_stats(data, ppc_data)
```

### Code cell 9

```python
ppc_data.head(10)
```

### Code cell 10

```python
ppc_compare = hddm.utils.post_pred_stats(data, ppc_data)
```

### Code cell 11

```python
ppc_compare.head(10)
```

### Code cell 12

```python
%%time
ppc_stats = hddm.utils.post_pred_stats(data, ppc_data, call_compare=False)
```

### Code cell 13

```python
ppc_stats.head(10)
```

## Using PPC for model comparison with the `groupby` argument

### Code cell 15

```python
m_pooled = hddm.HDDM(data, # v does not depend on conditions
                    include = ['v', 'a', 't', 'z'],
                        informative = True,
                        is_group_model = False) 
m_pooled.sample(1000, burn=20, save_name = "test/m_pooled")
ppc_data_pooled = hddm.utils.post_pred_gen(m_pooled, groupby=['condition'])
```

### Code cell 16

```python
hddm.utils.post_pred_stats(data, ppc_data_pooled)
```

## Defining your own summary statistics

### Code cell 18

```python
%%time
ppc_stats = hddm.utils.post_pred_stats(data, ppc_data, stats=lambda x: np.mean(x), call_compare=False)
```

### Code cell 19

```python
ppc_stats.head()
```

## Summary statistics relating to outside variables

### Code cell 21

```python
from numpy.random import randn
data['trlbytrl'] = randn(len(data))
```

### Code cell 22

```python
m_reg = hddm.HDDMRegressor(data, 'v ~ trlbytrl',
                          include = ['v', 'a', 't', 'z'],
                        informative = True,
                        is_group_model = False)
m_reg.sample(1000, burn=20, save_name = "test/m_reg")
```

### Code cell 23

```python
%%time
ppc_data = hddm.utils.post_pred_gen(m_reg, append_data=True)
ppc_data
```

### Code cell 24

```python
from scipy.stats import linregress
ppc_regression = []
for (node, sample), sim_data in ppc_data.groupby(level=(0, 1)):
    ppc_regression.append(linregress(sim_data.trlbytrl, sim_data.rt_sampled)[0]) # slope

orig_regression = linregress(data.trlbytrl, data.rt)[0]
```

### Code cell 25

```python
# cnt = 0
# for (node, sample), sim_data in ppc_data.groupby(level=(0, 1)):
#     print(sim_data)
#     cnt += 1
#     if cnt > 2:
#         break
```

### Code cell 26

```python
plt.hist(ppc_regression)
plt.axvline(orig_regression, c='r', lw=3)
plt.xlabel('slope')
```
