# tutorial post pred

> Converted from `hddm/hddm/examples/tutorial_post_pred.ipynb`. Code is preserved; rich outputs are omitted.

## Posterior Predictive Checks

In this tutorial you will learn how to run posterior predictive checks in HDDM.

A posterior predictive check is a very useful tool when you want to evaluate if your model can reproduce key patterns in your data. Specifically, you can define a summary statistic that describes the pattern you are interested in (e.g. accuracy in your task) and then simulate new data from the posterior of your fitted model. You can the apply the the summary statistic to each of the data sets you simulated from the posterior and see if the model does a good job of reproducing this pattern by comparing the summary statistics from the simulations to the summary statistic caluclated over the model.

What is critical is that you do not only get a single summary statistic from the simulations but a whole distribution which captures the uncertainty in our model estimate.

Lets do a simple analysis using simulated data. First, import HDDM.

### Code cell 3

```python
import hddm
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
```

Simulate data from known parameters and two conditions (easy and hard).

### Code cell 5

```python
data, params = hddm.generate.gen_rand_data(
    params={"easy": {"v": 1, "a": 2, "t": 0.3}, "hard": {"v": 1, "a": 2, "t": 0.3}}
)
```

First, lets estimate the same model that was used to generate the data.

### Code cell 7

```python
m = hddm.HDDM(data, depends_on={"v": "condition"})
m.sample(1000, burn=20)
```

Next, we'll want to simulate data from the model. By default, `post_pred_gen()` will use 500 parameter values from the posterior (i.e. posterior samples) and simulate a different data set for each parameter value.

### Code cell 9

```python
print(m.nodes_db)
```

### Code cell 10

```python
hddm.analyze.plot_posterior_nodes(m.nodes_db.loc[["v(easy)", "v(hard)"], "node"])
```

### Code cell 11

```python
ppc_data = hddm.utils.post_pred_gen(m)
```

### Code cell 12

```python
hddm.utils.post_pred_stats(data, ppc_data)
```

The returned data structure is a pandas `DataFrame` object with a hierarchical index.

### Code cell 14

```python
ppc_data.head(10)
```

The first level of the `DataFrame` contains each observed node. In this case the easy condition. If we had multiple subjects we would get one for each subject.

The second level contains the simulated data sets. Since we simulated 500, these will go from 0 to 499 -- each with generated from a different parameter value sampled from the posterior.

The third level is the same index as used in the data and numbers each trial in your data.

For more information on how to work with hierarchical indices, see the [Pandas documentation](http://pandas.pydata.org/pandas-docs/stable/indexing.html#hierarchical-indexing-multiindex).

There are also some helpful options like `append_data` you can pass to `post_pred_gen()`.

### Code cell 16

```python
help(hddm.utils.post_pred_gen)
```

Now we want to compute the summary statistics over each simulated data set and compare that to the summary statistic of our actual data by calling `post_pred_stats()`.

### Code cell 18

```python
ppc_compare = hddm.utils.post_pred_stats(data, ppc_data)
```

### Code cell 19

```python
print(ppc_compare)
```

As you can see, we did not have to define the summary statistics as by default, `HDDM` already calculates a bunch of useful statistics for RT analysis such as the accuracy, mean RT of the upper and lower boundary (ub and lb respectively), standard deviation and quantiles. These are listed in the rows of the DataFrame.

For each distribution of summary statistics there are multiple ways to compare them to the summary statistic obtained on the observerd data. These are listed in the columns. `observed` is just the value of the summary statistic of your data. `mean` is the mean of the summary statistics of the simulated data sets (they should be a good match if the model reproduces them). `std` is a measure of how much variation is produced in the summary statistic.

The rest of the columns are measures of how far the summary statistic of the data is away from the summary statistics of the simulated data. `SEM` = standard error from the mean, `MSE` = mean-squared error, `credible` = in the 95% credible interval.

Finally, we can also tell `post_pred_stats()` to return the summary statistics themselves by setting `call_compare=False`:

### Code cell 22

```python
ppc_stats = hddm.utils.post_pred_stats(data, ppc_data, call_compare=False)
```

### Code cell 23

```python
print(ppc_stats.head())
```

This `DataFrame` has a row for each simulated data set. The columns are the different summary statistics.

### Using PPC for model comparison with the `groupby` argument

One useful application of PPC is to perform model
comparison. Specifically, you might estimate two models, one for which
a certain parameter is split for a condition (say drift-rate ``v`` for
hard and easy conditions to stay with our example above) and one in
which those conditions are pooled and you only estimate one
drift-rate.

You then want to test which model explains the data better to assess
whether the two conditions are really different. To do this, we can
generate data from both models and see if the pooled model
systematically misses aspects of the RT data of the two
conditions. This is what the ``groupby`` keyword argument is
for. Without it, if you ran ``post_pred_gen()`` on the pooled model
you would get simulated RT data which was not split by
conditions. Note that while the RT data will be split by condition,
the exact same parameters are used to simulate data of the two
conditions as the pooled model does not separate them. It simply
allows us to match the two conditions present in the data to the
jointly simulated data more easily.

### Code cell 26

```python
m_pooled = hddm.HDDM(data)  # v does not depend on conditions
m_pooled.sample(1000, burn=20)
ppc_data_pooled = hddm.utils.post_pred_gen(m_pooled, groupby=["condition"])
```

You could then compare ``ppc_data_pooled`` to ``ppc_data`` above (by
passing them to ``post_pred_stats``) and find that the model with
separate drift-rates accounts for accuracy (``mean_ub``) in both
conditions, while the pooled model can't account for accuracy in
either condition (e.g. lower ``MSE``).

### Defining your own summary statistics

You can also define your own summary statistics and pass them to `post_pred_stats()`:

### Code cell 30

```python
ppc_stats = hddm.utils.post_pred_stats(
    data, ppc_data, stats=lambda x: np.mean(x), call_compare=False
)
```

### Code cell 31

```python
ppc_stats.head()
```

Note that `stats` can also be a dictionary mapping the name of the summary statistic to its function.

### Summary statistics relating to outside variables

Another useful way to apply posterior predictive checks is if you have trial-by-trial measure (e.g. EEG brain measure). In that case the `append_data` keyword argument is useful.

Lets add a dummy column to our data. This is going to be uncorrelated to anything but you'll get the idea.

### Code cell 35

```python
from numpy.random import randn

data["trlbytrl"] = randn(len(data))
```

### Code cell 36

```python
m_reg = hddm.HDDMRegressor(data, "v ~ trlbytrl")
m_reg.sample(1000, burn=20)

ppc_data = hddm.utils.post_pred_gen(m_reg, append_data=True)
```

### Code cell 37

```python
from scipy.stats import linregress

ppc_regression = []
for (node, sample), sim_data in ppc_data.groupby(level=(0, 1)):
    ppc_regression.append(
        linregress(sim_data.trlbytrl, sim_data.rt_sampled)[0]
    )  # slope

orig_regression = linregress(data.trlbytrl, data.rt)[0]
```

### Code cell 38

```python
cnt = 0
for (node, sample), sim_data in ppc_data.groupby(level=(0, 1)):
    print(sim_data)
    cnt += 1
    if cnt > 2:
        break
```

### Code cell 39

```python
plt.hist(ppc_regression)
plt.axvline(orig_regression, c="r", lw=3)
plt.xlabel("slope")
```

As you can see, the simulated data sets have on average no correlation to our trial-by-trial measure (just as in the data) but we also get a nice sense of the uncertainty in our estimation.
