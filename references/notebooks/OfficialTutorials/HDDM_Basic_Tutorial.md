# HDDM Basic Tutorial

> Converted from `OfficialTutorials/HDDM_Basic_Tutorial.ipynb`. Code is preserved; rich outputs are omitted.

### Code cell 1

```python
import pandas as pd
import matplotlib.pyplot as plt
```

### Code cell 2

```python
%matplotlib inline
import hddm
import kabuki
import arviz as az

print("The current version of kabuki is: ", kabuki.__version__)
print("The current version of HDDM is: ", hddm.__version__)
print("The current version of arviz is: ", az.__version__)
```

# Load data

### Code cell 4

```python
data = hddm.load_csv('/opt/conda/lib/python3.8/site-packages/hddm/examples/cavanagh_theta_nn.csv')
```

### Code cell 5

```python
data.head(10)
```

### Code cell 6

```python
import numpy as np
np.unique(data.subj_idx)
```

`subj_idx`: Subject ID, 14 subjects with index starting from 0  
`stim`: Condition based on pairs of stimuli. Win-win (WW), lose-lose (LL), win-lose (WL)  
`rt`: Response time in seconds  
`reponse`: Accuracy coded (1 = correct, more rewarding stimulus selected).  
`theta`: Theta band EEG activity.  
`dbs`: Deep brain simulation on/off.  
`conf`: Conflict condition. WL is easiest and low conflict (LC); the other two conditions are high conflict (HC).

Response times for all subjects

### Code cell 9

```python
data = hddm.utils.flip_errors(data)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_data in data.groupby('subj_idx'):
    subj_data.rt.hist(bins=20, histtype='step', ax=ax)
```

# Fit a hierarchical model

### Code cell 11

```python
%%time
# Instantiate model object passing it our data (no need to call flip_errors() before passing it).

# This will tailor an individual hierarchical DDM around your dataset.
m = hddm.HDDM(data, include = ['v', 'a', 't', 'z'],
                        informative = True,
                        is_group_model = True)
# find a good starting point which helps with the convergence.
m.find_starting_values()
# start drawing 2000 samples and discarding 20 as burn-in (usually you want to have a longer burn-in period)
m.sample(2000, burn=20)
```

### Code cell 12

```python
stats = m.gen_stats()
stats[stats.index.isin(['a', 'a_std', 'a_subj.0', 'a_subj.1'])]
```

### Code cell 13

```python
m.plot_posteriors(['a', 't', 'v', 'a_std'])
```

### Code cell 14

```python
%%time
models = []
for i in range(5):
    m = hddm.HDDM(data, include = ['v', 'a', 't', 'z'],
                        informative = True,
                        is_group_model = True)
    m.find_starting_values()
    m.sample(2000, burn=500)
    models.append(m)

rhats = hddm.analyze.gelman_rubin(models)
rhats = pd.DataFrame.from_dict(rhats, orient='index', columns=['R_hat']).sort_values('R_hat', ascending=False)
rhats.head(20)

# The below code use the new argument `chains` and it is more faster than the origin code above. 
# import arviz as az
# models = hddm.HDDM(
#     data, 
#     include = ['v', 'a', 't', 'z'],
#     informative = True,
#     is_group_model = True
# )
# models.find_starting_values()
# infData = models.sample(2000, chains = 500, InfData = True)
# az.summary(infData, kind = 'diagnostics')
```

### Code cell 15

```python
m.plot_posterior_predictive(figsize=(14, 10))
```

## Condition specific drift rates

### Code cell 17

```python
%%time
m_stim = hddm.HDDM(data, include = ['v', 'a', 't', 'z'],
                   informative = True,
                   is_group_model = True,
                   depends_on={'v': 'stim'})
m_stim.find_starting_values()
m_stim.sample(2000, burn=100)
```

### Code cell 18

```python
v_WW, v_LL, v_WL = m_stim.nodes_db.node[['v(WW)', 'v(LL)', 'v(WL)']]
hddm.analyze.plot_posterior_nodes([v_WW, v_LL, v_WL])
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior of drift-rate group means')
```

### Code cell 19

```python
print("P(WW > LL) = ", (v_WW.trace() > v_LL.trace()).mean())
print("P(LL > WL) = ", (v_LL.trace() > v_WL.trace()).mean())
```

### Code cell 20

```python
print("Lumped model DIC: %f" % m.dic)
print("Stimulus model DIC: %f" % m_stim.dic)
```

## Within-subject effects

### Code cell 22

```python
from patsy import dmatrix
dmatrix("C(stim, Treatment('WL'))", data.head(10))
```

### Code cell 23

```python
m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))", 
                                   include = ['v', 'a', 't', 'z'],
                                   informative = True,
                                   is_group_model = True)
```

### Code cell 24

```python
%%time
m_within_subj.sample(2000, burn=100)
```

### Code cell 25

```python
v_WL, v_LL, v_WW = m_within_subj.nodes_db.loc[["v_Intercept",
                                              "v_C(stim, Treatment('WL'))[T.LL]",
                                              "v_C(stim, Treatment('WL'))[T.WW]"], 'node']
hddm.analyze.plot_posterior_nodes([v_WL, v_LL, v_WW])
plt.xlabel('drift-rate')
plt.ylabel('Posterior probability')
plt.title('Group mean posteriors of within-subject drift-rate effects.')
```

## Fitting regression models

### Code cell 27

```python
m_reg = hddm.HDDMRegressor(data[data.dbs == 0],
                           "a ~ theta:C(conf, Treatment('LC'))",
                           depends_on={'v': 'stim'},
                           include = ['v', 'a', 't', 'z'],
                           informative = True,
                           is_group_model = True)
```

### Code cell 28

```python
%%time
m_reg.sample(2000, burn=100)
```

### Code cell 29

```python
m_reg_stats = m_reg.gen_stats()
# pd.set_option('display.max_rows', None)
m_reg_stats.head(20)
```

### Code cell 30

```python
theta = m_reg.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
hddm.analyze.plot_posterior_nodes([theta], bins=20)
plt.xlabel('Theta coeffecient in ')
print("P(a_theta < 0) = ", (theta.trace() < 0).mean())
```

## Dealing with outliers

### Code cell 32

```python
outlier_data, params = hddm.generate.gen_rand_data(params={'a': 2, 't': .4, 'v': .5},
                                                   size=200, n_fast_outliers=10)
```

### Code cell 33

```python
%%time
m_no_outlier = hddm.HDDM(outlier_data, p_outlier = 0.0,
                         include = ['v', 'a', 't', 'z'],
                         informative = True,
                         is_group_model = False)
m_no_outlier.sample(2000, burn=50)
```

### Code cell 34

```python
m_no_outlier.plot_posterior_predictive()
plt.title('Posterior predictive')
plt.xlabel('RT')
plt.ylabel('Probability density')
```

### Code cell 35

```python
%%time
m_outlier = hddm.HDDM(outlier_data, p_outlier=.05,
                      include = ['v', 'a', 't', 'z'],
                      informative = True,
                      is_group_model = False)
m_outlier.sample(2000, burn=50)
```

### Code cell 36

```python
m_outlier.plot_posterior_predictive()
plt.title('Posterior predictive')
plt.xlabel('RT')
plt.ylabel('Probability density')
```
