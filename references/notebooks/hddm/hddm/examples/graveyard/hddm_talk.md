# hddm talk

> Converted from `hddm/hddm/examples/graveyard/hddm_talk.ipynb`. Code is preserved; rich outputs are omitted.

<center><h1> HDDM: Hierarchical Bayesian Estimation of the Drift Diffusion model</h1><p>
<h3>Thomas Wiecki</h3><p>
<h3>Former: PhD at Brown University on Computational Psychiatry</h3><p>
<h3>Current: Lead Data Scientist at Quantopian Inc.</h3>

## Contents

* Features
* Installation
* First model
* Group-wise conditions
* Trial-by-trial effects
* Outliers

## Why should you use it?

## Bayesian parameter estimation -> Posteriors over parameters.

## It's hierarchical.

<img src="graphical_hddm.svg"/>

## Informative priors based on literature

<img src="hddm_info_priors.svg"/>

## ... all lead to better parameter recovery

<img src="http://www.frontiersin.org/files/Articles/55610/fninf-07-00014-HTML/image_m/fninf-07-00014-g006.jpg">

## Other reasons
* Heavily optimized likelihoods for speed (minutes to couple of hours for complex models).
* Tuned samplers (slice sampling) for fast convergence.
* Trial-by-trial regressions allow estimation of influence of brain measures onto parameters.
* Free & Open-source (BSD license)
* Python (not Matlab)
* Good software engineering practices (unittests, continuous integration)

## It's Roger Ratcliff approved


"We found that the hierarchical diffusion method [as implemented by HDDM] performed very well, and is the method of choice when the number of observations is small."<br>

Roger Ratcliff, grandfather of the DDM, in a paper comparing all available tools to do DDM analysis. </center>

## Prof. James Rowe (Cambridge University)

"The HDDM modelling gave insights into the effects of disease that were simply not visible from a traditional analysis of RT/Accuracy. It provides a clue as to why many disorders including PD and PSP can give the paradoxical combination of akinesia and impulsivity. Perhaps of broader interest, the hierarchical drift diffusion model turned out to be very robust. In separate work, we have found that the HDDM gave accurate estimates of decision parameters with many fewer than 100 trials, in contrast to the hundreds or even thousands one might use for ‘traditional’ DDMs. This meant it was realistic to study patients who do not tolerate long testing sessions."

## Installation

* Install the Anaconda Python distribution from Continuum. Available for all platforms.
* Type: `conda install -c pymc hddm`

## First steps

### Code cell 13

```python
%matplotlib inline
```

## Importing the modules

### Code cell 15

```python
import pandas as pd  # Input, output and process tabular data
import matplotlib.pyplot as plt  # Plotting
import hddm  # Our toolbox

print(hddm.__version__)
```

### Loading data from csv

### Code cell 17

```python
!head cavanagh_theta_nn.csv
```

### We use the ``hddm.load_csv()`` function to load this file.

### Code cell 19

```python
data = hddm.load_csv("./cavanagh_theta_nn.csv")
```

### This is what it looks like

### Code cell 21

```python
data.head(12)
```

### Plotting RT distributions

### Code cell 23

```python
data = hddm.utils.flip_errors(data)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="RT", ylabel="count", title="RT distributions")
for i, subj_data in data.groupby("subj_idx"):
    subj_data.rt.hist(bins=20, histtype="step", ax=ax)
```

### Fitting a hierarchical model

### Code cell 25

```python
# Instantiate model object passing it our data (no need to call flip_errors() before passing it).
# This will tailor an individual hierarchical DDM around your dataset.
m = hddm.HDDM(data)

# find a good starting point which helps with the convergence.
m.find_starting_values()

# start drawing 2000 samples and discarding 20 as burn-in
m.sample(2000, burn=20)
```

## Generating summary statistics

### Code cell 27

```python
m.gen_stats()[["mean"]]
```

## Plotting the posterior

### Code cell 29

```python
m.plot_posteriors(["a", "t", "v", "a_std"])
```

## How well does model fit data? -> Posterior predictive plot

### Code cell 31

```python
m.plot_posterior_predictive(figsize=(14, 10))
```

## Defining conditions with `depends_on`

### Code cell 33

```python
m_stim = hddm.HDDM(data, depends_on={"v": "stim"})
m_stim.find_starting_values()
m_stim.sample(10000, burn=1000)
```

## Comparing drift-rates across conditions

### Code cell 35

```python
v_WW, v_LL, v_WL = m_stim.nodes_db.node[["v(WW)", "v(LL)", "v(WL)"]]
hddm.analyze.plot_posterior_nodes([v_WW, v_LL, v_WL])
plt.xlabel("drift-rate")
plt.ylabel("Posterior probability")
plt.title("Posterior of drift-rate group means")
```

## Hypothesis testing

### Code cell 37

```python
print("P(v_WW > v_LL) = {:.3f}%".format((v_WW.trace() > v_LL.trace()).mean() * 100))
print("P(v_LL > v_WL) = {:.3f}%".format((v_LL.trace() > v_WL.trace()).mean() * 100))
```

## Model comparison using DIC

* Deviance Information Criterion.
* Measure trading off model fit and model complexity.
* Not perfect but useful and easy to compute.
* Lower is better.

### Code cell 39

```python
print("Lumped model DIC = %f" % m.dic)
print("Stimulus model DIC = %f" % m_stim.dic)
```

## Model comparison using Posterior Predictive checks

* Generate data sets from model's posterior.
* Compare generated data sets to original data to assess if key patterns are reproduced by the model.
* See [http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html](http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html) for more detail of how to do this in HDDM.

## Within-subject effects

## Specify a glm using R-like syntax with `patsy`

### Code cell 43

```python
from patsy import dmatrix

dmatrix("C(stim)", data.head(10))
```

## Pass glm-descriptor to `HDDMRegressor`

### Code cell 45

```python
m_within_subj = hddm.HDDMRegressor(data, "v ~ C(stim, Treatment('WL'))")
```

### Code cell 46

```python
m_within_subj.sample(5000, burn=200)
```

### Code cell 47

```python
v_WL, v_LL, v_WW = m_within_subj.nodes_db.ix[
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
```

## Fitting regression models

* Main effects of theta and dbs, as well as theta x dbs interaction.
* Within-subject effect of conflict on threshold.
* For more information, see http://ski.clps.brown.edu/papers/Cavanagh_DBSEEG.pdf

### Code cell 49

```python
m_reg = hddm.HDDMRegressor(data, "a ~ theta*dbs + C(conf)", depends_on={"v": "stim"})
```

### Code cell 50

```python
m_reg.sample(5000, burn=200)
```

### Code cell 51

```python
theta = m_reg.nodes_db.node["a_theta:C(conf, Treatment('LC'))[HC]"]
hddm.analyze.plot_posterior_nodes([theta], bins=20)
plt.xlabel("Theta coeffecient in ")
print("P(a_theta < 0) = ", (theta.trace() < 0).mean())
```

## Outliers are a fact of life

### Code cell 53

```python
outlier_data, params = hddm.generate.gen_rand_data(
    params={"a": 2, "t": 0.4, "v": 0.5}, size=200, n_fast_outliers=10
)
```

### Code cell 54

```python
m_no_outlier = hddm.HDDM(outlier_data)
m_no_outlier.sample(2000, burn=50)
```

## Fit is strongly affected, especially by fast outliers

### Code cell 56

```python
m_no_outlier.plot_posterior_predictive()
plt.title("Posterior predictive")
plt.xlabel("RT")
plt.ylabel("Probability density")
```

## Robustness to outliers with `p_outlier`

### Code cell 58

```python
m_outlier = hddm.HDDM(outlier_data, p_outlier=0.05)
m_outlier.sample(2000, burn=20)
```

### Code cell 59

```python
m_outlier.plot_posterior_predictive()
plt.title("Posterior predictive")
plt.xlabel("RT")
plt.ylabel("Probability density")
```

# Questions?

## Links

* Documentation: [http://ski.clps.brown.edu/hddm_docs/](http://ski.clps.brown.edu/hddm_docs/)
* Code: https://github.com/hddm-devs/hddm
* More info on posterior predictive checks: [http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html](http://ski.clps.brown.edu/hddm_docs/tutorial_post_pred.html)
