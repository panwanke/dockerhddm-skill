# demo RLHDDMtutorial

> Converted from `hddm/hddm/examples/demo_RLHDDMtutorial.ipynb`. Code is preserved; rich outputs are omitted.

# Tutorial for analyzing instrumental learning data with the HDDMrl module
This is a tutorial for using the HDDMrl module to simultaneously estimate reinforcement learning parameters and decision parameters within a fully hierarchical bayesian estimation framework, including steps for sampling, assessing convergence, model fit, parameter recovery, and posterior predictive checks (model validation). The module uses the reinforcement learning drift diffusion model (RLDDM), a reinforcement learning model that replaces the standard “softmax” choice function with a drift diffusion process. The softmax and drift diffusion process is equivalent for capturing choice proportions, but the DDM also takes RT distributions into account; options are provided to also only fit RL parameters without RT. The RLDDM estimates trial-by-trial drift rate as a scaled difference in expected rewards (expected reward for upper bound alternative minus expected reward for lower bound alternative). Expected rewards are updated with a delta learning rule using either a single learning rate or with separate learning rates for positive and negative prediction errors. The model also includes the standard DDM-parameters. The RLDDM is described in detail in [Pedersen, Frank & Biele (2017).](http://ski.clps.brown.edu/papers/PedersenEtAl_RLDDM.pdf) (Note this approach differs from Frank et al (2015) who used HDDM to fit instrumental learning but did not allow for simultaneous estimation of learning parameters). 

## OUTLINE 

[1. Background](#1.-Background) <br>
[2. Installing the module](#2.-Installing-the-module) <br>
[3. How the RLDDM works](#3.-How-the-RLDDM-works) <br>
[4. Structuring data](#4.-Structuring-data) <br>
[5. Running basic model](#5.-Running-basic-model) <br>
[6. Checking results](#6.-Checking-results) <br>
[7. Posterior predictive checks](#7.-Posterior-predictive-checks)<br>
[8. Parameter recovery](#8.-Parameter-recovery)<br>
[9. Separate learning rates for positive and negative prediction errors](#9.-Separate-learning-rates-for-positive-and-negative-prediction-errors)<br>
[10. depends_on vs. split_by](#10.-depends_on-vs.-split_by)<br>
[11. Probabilistic binary outcomes vs. normally distributed outcomes](#11.-Probabilistic-binary-outcomes-vs.-normally-distributed-outcomes)<br>
[12. HDDMrlRegressor](#12.-HDDMrlRegressor)<br>
[13. Regular RL without RT](#13.-Regular-RL-without-RT)

## 1. Background
Traditional RL models typically assume static decision processes (e.g. softmax), and the DDM typically assumes static decision variables (stimuli are modeled with the same drift rate across trials). The RLDDM combines dynamic decision variables from RL and dynamic choice process from DDM by assuming trial-by-trial drift rate that depends on the difference in expected rewards, which are updated on each trial by a rate of the prediction error dependent on the learning rate. The potential benefit of the RLDDM is thus to gain a better insight into decision processes in instrumental learning by also accounting for speed of decision making.

## 2. Installing the module
The new version of HDDM (version 0.7.1) that includes the RLDDM is currently not uploaded to conda. So to install you would either have to use pip:<br>
'pip install hddm' <br>
OR <br>
docker: 'pull madslupe/hddm', which runs hddm in jupyter notebook <br>

## 3. How the RLDDM works
The main idea of the RLDDM is that reward-based choices can be captured by an accumulation-to-bound process where drift rate is proportional to the difference in expected reward between options, and where expected rewards subsequently are updated in a trial-by-trial basis via reinforcement learning. <br><br>
__drift rate on each trial depends on difference in expected rewards for the two alternatives (q_up and q_low):__ <br>
drift rate = (q_up - q_low) * scaling <br><br>
_where the scaling parameter describes the weight to put on the difference in q-values._<br><br>
__expeceted reward (q) for chosen option is updated according to delta learning rule :__ <br>
q_chosen = q_chosen + alpha * (feedback-q_chosen) <br><br>
_where alpha weights the rate of learning on each trial._<br><br>
So in principle all you need is the Wiener first passage time likelihood-function. The reason why HDDM is useful (and hence also HDDMrl) is that it automates the process of setting up and running your model, which tends to be very time consuming. So after structuring the data it is simple to run a model with HDDMrl. In particular it separates subjects and conditions (using the split_by-column, see next section) so that the updating process works correctly, which can be especially difficult to do for RL models.

## 4. Structuring data
The HDDMrl module was created to make it easier to model instrumental learning data with the RLDDM. If you are familiar with using HDDM it shouldn't be a big step to start using HDDMrl. Please refresh your memory by starting with the [tutorial for HDDM](http://ski.clps.brown.edu/hddm_docs/index.html) first (especially critical if you have not used HDDM at all). Running HDDMrl does require a few extra steps compared to HDDM, and because the model includes increased potential for parameter colinearity  (typically learning rate and the scaling parameter on drift rate) it is even more important to assess model fit, which will be covered below. Here are the most important steps for structuring your dataframe:
1. Sort trials in ascending order within subject and condition, to ensure proper updating of expected rewards.
2. Include a column called __'split_by'__ which identifies the different task conditions (__as integers__), to ensure reward updating will work properly for each condition without mixing values learned from one trial type to another. 
3. Code the response column with [stimulus-coding] (http://ski.clps.brown.edu/hddm_docs/howto.html#code-subject-responses). Although stimulus-coding and accuracy-coding often are the same in instrumental learning it is important that the upper and lower boundaries are represented by the same alternative within a condition, because expected rewards are linked to the thresholds/boundaries.
4. __feedback__-column. This should be the reward received for the chosen option on each trial.
5. __q_init__. Adjusting initial q-values is something that can improve model fit quite a bit. To allow the user to set their own initial values we therefore require that the dataframe includes a column called q_init. The function will set all initial q-values according to the first value in q_init. So this is not the most elegant method of allowing users to set inital value for expected rewards, but it works for now.

### Required columns in data:
* __rt__: in seconds, same as in HDDM
* __response__: 0 or 1. defines chosen stimulus, not accuracy.
* __split_by__: needs to be an integer. Split_by defines conditions (trial-types) that should have the same variables (e.g. Q values) within subject: the trials need to be split by condition to ensure proper updating of expected rewards that do not bleed over into other trial-types. (e.g. if you have stimulus A and get reward you want that updated value to impact choice only for the next stimulus A trial but not necessarily the immediate trial afterwards, which may be of a different condition) 
* __subj_idx__: same as in HDDM, but even more important here because it is used to split trials
* __feedback__: feedback on the current trial. can be any value.
* __q_init__: used to initialize expected rewards. can be any value, but an unbiased initial value should be somewhere between the minimum and and maximum reward values (e.g. 0.5 for tasks with rewards of 0 and 1).

## 5. Running basic model
To illustrate how to run the model we will use example data from the learning phase of the probabilistic selection task (PST). During the learning phase of the PST subjects choose between two  stimuli presented as Hiragana-letters (here represented as letters from the latin alphabet). There are three conditions with different probabilities of receiving reward (feedback=1) and non-reward (feedback=0). In the AB condition A is rewarded with 80% probability, B with 20%. In the CD condition C is rewarded with 70% probability and D with 30%, while in the EF condition E is rewarded with a 60% probability and F with 40%. The dataset is included in the data-folder in your installation of HDDM.

### Code cell 7

```python
# import
import pandas as pd
import numpy as np
import hddm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pymc
import kabuki

sns.set(style="white")
%matplotlib inline
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
```

### Code cell 8

```python
# load data. you will find this dataset in your hddm-folder under hddm/examples/rlddm_data.csv
data = hddm.load_csv("rlddm_data.csv")
# check structure
data.head()
```

The columns in the datafile represent: __subj_idx__ (subject id), __response__ (1=best option), __cond__ (identifies condition, but not used in model), __rt__ (in seconds), 0=worst option), __trial__ (the trial-iteration for a subject within each condition), __split_by__ (identifying condition, used for running the model), __feedback__ (whether the response given was rewarded or not), __q_init__ (the initial q-value used for the model, explained above).

### Code cell 10

```python
# run the model by calling hddm.HDDMrl (instead of hddm.HDDM for normal HDDM)
m = hddm.HDDMrl(data)
# set sample and burn-in
m.sample(1500, burn=500, dbname="traces.db", db="pickle")
# print stats to get an overview of posterior distribution of estimated parameters
m.print_stats()
```

__Interpreting output from print_stats:__  <br>
The model estimates group mean and standard deviation parameters and subject parameters for the following latent variables: <br>
a = decision threshold <br>
v = scaling parameter <br>
t = non-decision time <br>
alpha = learning rate, note that it's not bound between 0 and 1. to transform take inverse logit: np.exp(alpha)/(1+np.exp(alpha)) <br>
The columns represent the mean, standard deviation and quantiles of the approximated posterior distribution of each parameter

### HDDMrl vs. HDDM
__There are a few things to note that is different from the normal HDDM model.__ <br>
First of all, the estimated learning rate does not necessarily fall between 0 and 1. This is because it is estimated as a normal distribution for purposes of sampling hierarchically and then transformed by an inverse logit function to 0<alpha<1. So to interpret alpha as learning rate you have to transform the samples in the trace back with np.exp(alpha)/(1+np.exp(alpha)). And if you estimate separate learning rates for positive and negative prediction errors ([see here](#9.-Separate-learning-rates-for-positive-and-negative-prediction-errors)) then you get learning rate for negative prediction errors with np.exp(alpha)/(1+np.exp(alpha)) and positive prediction errors with np.exp(pos_alpha)/(1+np.exp(pos_alpha)).<br>
Second, the v-parameter in the output is the scaling factor that is multiplied by the difference in q-values, so it is not the actual drift rate (or rather, it is the equivalent drift rate when the difference in Q values is exactly 1).

## 6. Checking results

### Code cell 14

```python
# plot the posteriors of parameters
m.plot_posteriors()
```

__Fig__. The mixing of the posterior distribution and autocorrelation looks ok.

### Convergence of chains
The Gelman-Rubin statistic is a test of whether the chains in the model converges. The Gelman-Ruben statistic measures the degree of variation between and within chains. Values close to 1 indicate convergence and that there is small variation between chains, i.e. that they end up as the same distribution across chains. A common heuristic is to assume convergence if all values are below 1.1. To run this you need to run multiple models, combine them and perform the Gelman-Rubin statistic:

### Code cell 17

```python
# estimate convergence
from kabuki.analyze import gelman_rubin

models = []
for i in range(3):
    m = hddm.HDDMrl(data=data)
    m.sample(1500, burn=500, dbname="traces.db", db="pickle")
    models.append(m)

gelman_rubin(models)
```

### Code cell 18

```python
np.max(list(gelman_rubin(models).values()))
```

The model seems to have converged, i.e. the Gelman-Rubin statistic is below 1.1 for all parameters. It is important to always run this test, especially for more complex models ([as with separate learning rates for positive and negative prediction errors](#9.-Separate-learning-rates-for-positive-and-negative-prediction-errors)). So now we can combine these three models to get a better approximation of the posterior distribution.

### Code cell 20

```python
# Combine the models we ran to test for convergence.
m = kabuki.utils.concat_models(models)
```

### Joint posterior distribution
Another test of the model is to look at collinearity. If the estimation of parameters is very codependent (correlation is strong) it can indicate that their variance trades off, in particular if there is a negative correlation. The following plot shows there is generally low correlation across all combinations of parameters. It does not seem to be the case for this dataset, but common for RLDDM is a negative correlation between learning rate and the scaling factor, similar to what's usually observed between learning rate and inverse temperature for RL models that uses softmax as the choice rule (e.g. [Daw, 2011](https://www.oxfordscholarship.com/view/10.1093/acprof:oso/9780199600434.001.0001/acprof-9780199600434-chapter-001)).

### Code cell 22

```python
alpha, t, a, v = m.nodes_db.node[["alpha", "t", "a", "v"]]
samples = {"alpha": alpha.trace(), "t": t.trace(), "a": a.trace(), "v": v.trace()}
samp = pd.DataFrame(data=samples)


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), xy=(0.1, 0.9), xycoords=ax.transAxes)


g = sns.PairGrid(samp, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
g.savefig("matrix_plot.png")
```

## 7. Posterior predictive checks

An important test of the model is its ability to recreate the observed data. This can be tested with posterior predictive checks, which involves simulating data using estimated parameters and comparing observed and simulated results.

### extract traces
The first step then is to extract the traces from the estimated model. The function get_traces() gives you the samples (row) from the approaximated posterior distribution for all of the estimated group and subject parameters (column).

### Code cell 26

```python
traces = m.get_traces()
traces.head()
```

### simulating data
__Now that we have the traces the next step is to simulate data using the estimated parameters. The RLDDM includes a function to simulate data. Here's an example of how to use the simulation-function for RLDDM. This example explains how to generate data with binary outcomes. See [here](#11.-Probabilistic-binary-outcomes-vs.-normally-distributed-outcomes) for an example on simulating data with normally distributed outcomes. Inputs to function: <br>
a__ = decision threshold <br>
**t** = non-decision time <br>
__alpha__ = learning rate <br>
__pos_alpha__ = defaults to 0. if given it defines the learning rate for positive prediction errors. alpha then becomes the learning rate_ for negative prediction errors.  <br>
__scaler__ = the scaling factor that is multiplied with the difference in q-values to calculate trial-by-trial drift rate <br>
__p_upper__ = the probability of reward for the option represented by the upper boundary. The current version thus only works for outcomes that are either 1 or 0 <br>
__p_lower__ = the probability of reward for the option represented by the lower boundary. <br>
__subjs__ = number of subjects to simulate data for. <br>
__split_by__ = define the condition which makes it easier to append data from different conditions. <br>
__size__ = number of trials per subject. <br>

### Code cell 28

```python
hddm.generate.gen_rand_rlddm_data(
    a=1,
    t=0.3,
    alpha=0.2,
    scaler=2,
    p_upper=0.8,
    p_lower=0.2,
    subjs=1,
    split_by=0,
    size=10,
)
```

__How to interpret columns in the resulting dataframe__ <br>
__q_up__ = expected reward for option represented by upper boundary <br>
__q_low__ = expected reward for option represented by lower boundary <br>
__sim_drift__ = the drift rate for each trial calculated as: (q_up-q_low)*scaler <br>
__response__ = simulated choice <br>
__rt__ = simulated response time <br>
__feedback__ = observed feedback for chosen option <br>
__subj_idx__ = subject id (starts at 0) <br>
__split_by__ = condition as integer <br>
__trial__ = current trial (starts at 1) <br>

### Simulate data with estimated parameter values and compare to observed data
Now that we know how to extract traces and simulate data we can combine this to create a dataset similar to our observed data. This process is currently not automated but the following is an example code using the dataset we analyzed above.

### Code cell 31

```python
from tqdm import tqdm  # progress tracker

# create empty dataframe to store simulated data
sim_data = pd.DataFrame()
# create a column samp to be used to identify the simulated data sets
data["samp"] = 0
# load traces
traces = m.get_traces()
# decide how many times to repeat simulation process. repeating this multiple times is generally recommended,
# as it better captures the uncertainty in the posterior distribution, but will also take some time
for i in tqdm(range(1, 51)):
    # randomly select a row in the traces to use for extracting parameter values
    sample = np.random.randint(0, traces.shape[0] - 1)
    # loop through all subjects in observed data
    for s in data.subj_idx.unique():
        # get number of trials for each condition.
        size0 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 0)].trial.unique()
        )
        size1 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 1)].trial.unique()
        )
        size2 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 2)].trial.unique()
        )
        # set parameter values for simulation
        a = traces.loc[sample, "a_subj." + str(s)]
        t = traces.loc[sample, "t_subj." + str(s)]
        scaler = traces.loc[sample, "v_subj." + str(s)]
        alphaInv = traces.loc[sample, "alpha_subj." + str(s)]
        # take inverse logit of estimated alpha
        alpha = np.exp(alphaInv) / (1 + np.exp(alphaInv))
        # simulate data for each condition changing only values of size, p_upper, p_lower and split_by between conditions.
        sim_data0 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            size=size0,
            p_upper=0.8,
            p_lower=0.2,
            split_by=0,
        )
        sim_data1 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            size=size1,
            p_upper=0.7,
            p_lower=0.3,
            split_by=1,
        )
        sim_data2 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            size=size2,
            p_upper=0.6,
            p_lower=0.4,
            split_by=2,
        )
        # append the conditions
        sim_data0 = sim_data0.append([sim_data1, sim_data2], ignore_index=True)
        # assign subj_idx
        sim_data0["subj_idx"] = s
        # identify that these are simulated data
        sim_data0["type"] = "simulated"
        # identify the simulated data
        sim_data0["samp"] = i
        # append data from each subject
        sim_data = sim_data.append(sim_data0, ignore_index=True)
# combine observed and simulated data
ppc_data = data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "samp"]
].copy()
ppc_data["type"] = "observed"
ppc_sdata = sim_data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "type", "samp"]
].copy()
ppc_data = ppc_data.append(ppc_sdata)
ppc_data.to_csv("ppc_data_tutorial.csv")
```

### Plotting
Now that we have a dataframe with both observed and simulated data we can plot to see whether the simulated data are able to capture observed choice and reaction times. To capture the uncertainty in the simulated data we want to identify how much choice and reaction differs across the simulated data sets. A good measure of this is to calculate the highest posterior density/highest density interval for summary scores of the generated data. Below we calculate highest posterior density with an alpha set to 0.1, which means that we are describing the range of the 90% most likely values.

### Code cell 33

```python
# for practical reasons we only look at the first 40 trials for each subject in a given condition
plot_ppc_data = ppc_data[ppc_data.trial < 41].copy()
```

### Choice

### Code cell 35

```python
# bin trials to for smoother estimate of response proportion across learning
plot_ppc_data["bin_trial"] = pd.cut(
    plot_ppc_data.trial, 11, labels=np.linspace(0, 10, 11)
).astype("int64")
# calculate means for each sample
sums = (
    plot_ppc_data.groupby(["bin_trial", "split_by", "samp", "type"])
    .mean()
    .reset_index()
)
# calculate the overall mean response across samples
ppc_sim = sums.groupby(["bin_trial", "split_by", "type"]).mean().reset_index()
# initiate columns that will have the upper and lower bound of the hpd
ppc_sim["upper_hpd"] = 0
ppc_sim["lower_hpd"] = 0
for i in range(0, ppc_sim.shape[0]):
    # calculate the hpd/hdi of the predicted mean responses across bin_trials
    hdi = pymc.utils.hpd(
        sums.response[
            (sums["bin_trial"] == ppc_sim.bin_trial[i])
            & (sums["split_by"] == ppc_sim.split_by[i])
            & (sums["type"] == ppc_sim.type[i])
        ],
        alpha=0.1,
    )
    ppc_sim.loc[i, "upper_hpd"] = hdi[1]
    ppc_sim.loc[i, "lower_hpd"] = hdi[0]
# calculate error term as the distance from upper bound to mean
ppc_sim["up_err"] = ppc_sim["upper_hpd"] - ppc_sim["response"]
ppc_sim["low_err"] = ppc_sim["response"] - ppc_sim["lower_hpd"]
ppc_sim["model"] = "RLDDM_single_learning"
ppc_sim.to_csv("ppc_choicedata_tutorial.csv")
```

### Code cell 36

```python
# plotting evolution of choice proportion for best option across learning for observed and simulated data.
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True)
for i in range(0, 3):
    ax = axs[i]
    d = ppc_sim[(ppc_sim.split_by == i) & (ppc_sim.type == "simulated")]
    ax.errorbar(
        d.bin_trial,
        d.response,
        yerr=[d.low_err, d.up_err],
        label="simulated",
        color="orange",
    )
    d = ppc_sim[(ppc_sim.split_by == i) & (ppc_sim.type == "observed")]
    ax.plot(d.bin_trial, d.response, linewidth=3, label="observed")
    ax.set_title("split_by = %i" % i, fontsize=20)
    ax.set_ylabel("mean response")
    ax.set_xlabel("trial")
plt.legend()
fig.savefig("PPCchoice.pdf")
```

__Fig.__ The plots display the rate of choosing the best option (response = 1) across learning and condition. The model generates data (orange) that closely follows the observed behavior (blue), with the exception of overpredicting performance early in the most difficult condition (split_by=2). Uncertainty in the generated data is captured by the 90% highest density interval of the means across simulated datasets.

### RT

### Code cell 39

```python
# set reaction time to be negative for lower bound responses (response=0)
plot_ppc_data["reaction time"] = np.where(
    plot_ppc_data["response"] == 1, plot_ppc_data.rt, 0 - plot_ppc_data.rt
)
# plotting evolution of choice proportion for best option across learning for observed and simulated data. We use bins of trials because plotting individual trials would be very noisy.
g = sns.FacetGrid(plot_ppc_data, col="split_by", hue="type")
g.map(sns.kdeplot, "reaction time", bw=0.05).set_ylabels("Density")
g.add_legend()
g.savefig("PPCrt_dist.pdf")
```

__Fig.__ Density plots of observed and predicted reaction time across conditions. RTs for lower boundary choices (i.e. worst option choices) are set to be negative (0-RT) to be able to separate upper and lower bound responses.

## 8. Parameter recovery
To validate the RLDDM we ran a parameter recovery study to test to which degree the model can recover the parameter values used to simulate data. To do this we generated 81 synthetic datasets with 50 subjects performing 70 trials each. The 81 datasets were simulated using all combinations of three plausible parameter values for decision threshold, non-decision time, learning rate and the scaling parameter onto drift rate.

### Estimated values split by simulated vales 
We can plot simulated together with the estimated values to test the models ability to recover parameters, and to see if there are any values that are more difficult to recover than others.

### Code cell 43

```python
param_recovery = hddm.load_csv("recovery_sim_est_rlddm.csv")
```

### Code cell 44

```python
g = sns.catplot(x="a", y="e_a", data=param_recovery, palette="Set1")
g.set_axis_labels("Simulated threshold", "Estimated threshold")
plt.title("Decision threshold")
g.savefig("Threshold_recovery.pdf")
```

### Code cell 45

```python
g = sns.catplot(x="alpha", y="e_alphaT", data=param_recovery, palette="Set1")
g.set_axis_labels("Simulated alpha", "Estimated alpha")
plt.title("Learning rate")
g.savefig("Alpha_recovery.pdf")
```

### Code cell 46

```python
g = sns.catplot(x="scaler", y="e_v", data=param_recovery, palette="Set1")
g.set_axis_labels("Simulated scaling", "Estimated scaling")
plt.title("Scaling drift rate")
g.savefig("Scaler_recovery.pdf")
```

### Code cell 47

```python
g = sns.catplot(x="t", y="e_t", data=param_recovery, palette="Set1")
g.set_axis_labels("Simulated NDT", "Estimated NDT")
plt.title("Non-decision time")
g.savefig("NDT_recovery.pdf")
```

__Fig.__ The correlation between simulated and estimated parameter values are high, which means recovery is good. There is somewhat worse recovery for the learning rate and scaling parameter, which makes sense given that they to a degree can explain the same variance (see below).

## 9. Separate learning rates for positive and negative prediction errors
Several studies have reported differences in updating of expected rewards following positive and negative prediction errors (e.g. to capture differences between D1 and D2 receptor function). To model asymmetric updating rates for positive and negative prediction errors you can set dual=True in the model. This will produce two estimated learning rates; alpha and pos_alpha, of which alpha then becomes the estimated learning rate for negative prediction errors.

### Code cell 50

```python
# set dual=True to model separate learning rates for positive and negative prediction errors.
m_dual = hddm.HDDMrl(data, dual=True)
# set sample and burn-in
m_dual.sample(1500, burn=500, dbname="traces.db", db="pickle")
# print stats to get an overview of posterior distribution of estimated parameters
m_dual.print_stats()
```

### Code cell 51

```python
m_dual.plot_posteriors()
```

__Fig.__ There's more autocorrelation in this model compared to the one with a single learning rate. First, let's test whether it converges.

### Code cell 53

```python
# estimate convergence
models = []
for i in range(3):
    m = hddm.HDDMrl(data=data, dual=True)
    m.sample(1500, burn=500, dbname="traces.db", db="pickle")
    models.append(m)

# get max gelman-statistic value. shouldn't be higher than 1.1
np.max(list(gelman_rubin(models).values()))
```

### Code cell 54

```python
gelman_rubin(models)
```

Convergence looks good, i.e. no parameters with gelman-rubin statistic > 1.1.

### Code cell 56

```python
# Create a new model that has all traces concatenated
# of individual models.
m_dual = kabuki.utils.concat_models(models)
```

And then we can have a look at the joint posterior distribution:

### Code cell 58

```python
alpha, t, a, v, pos_alpha = m_dual.nodes_db.node[["alpha", "t", "a", "v", "pos_alpha"]]
samples = {
    "alpha": alpha.trace(),
    "pos_alpha": pos_alpha.trace(),
    "t": t.trace(),
    "a": a.trace(),
    "v": v.trace(),
}
samp = pd.DataFrame(data=samples)


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), xy=(0.1, 0.9), xycoords=ax.transAxes)


g = sns.PairGrid(samp, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
g.savefig("matrix_plot.png")
```

__Fig.__ The correlation between parameters is generally low.

## Posterior predictive check
The DIC for this dual learning rate model is better than for the single learning rate model. We can therefore check whether we can detect this improvement in the ability to recreate choice and RT patterns:

### Code cell 61

```python
# create empty dataframe to store simulated data
sim_data = pd.DataFrame()
# create a column samp to be used to identify the simulated data sets
data["samp"] = 0
# get traces, note here we extract traces from m_dual
traces = m_dual.get_traces()
# decide how many times to repeat simulation process. repeating this multiple times is generally recommended as it better captures the uncertainty in the posterior distribution, but will also take some time
for i in tqdm(range(1, 51)):
    # randomly select a row in the traces to use for extracting parameter values
    sample = np.random.randint(0, traces.shape[0] - 1)
    # loop through all subjects in observed data
    for s in data.subj_idx.unique():
        # get number of trials for each condition.
        size0 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 0)].trial.unique()
        )
        size1 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 1)].trial.unique()
        )
        size2 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 2)].trial.unique()
        )
        # set parameter values for simulation
        a = traces.loc[sample, "a_subj." + str(s)]
        t = traces.loc[sample, "t_subj." + str(s)]
        scaler = traces.loc[sample, "v_subj." + str(s)]
        # when generating data with two learning rates pos_alpha represents learning rate for positive prediction errors and alpha for negative prediction errors
        alphaInv = traces.loc[sample, "alpha_subj." + str(s)]
        pos_alphaInv = traces.loc[sample, "pos_alpha_subj." + str(s)]
        # take inverse logit of estimated alpha and pos_alpha
        alpha = np.exp(alphaInv) / (1 + np.exp(alphaInv))
        pos_alpha = np.exp(pos_alphaInv) / (1 + np.exp(pos_alphaInv))
        # simulate data for each condition changing only values of size, p_upper, p_lower and split_by between conditions.
        sim_data0 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size0,
            p_upper=0.8,
            p_lower=0.2,
            split_by=0,
        )
        sim_data1 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size1,
            p_upper=0.7,
            p_lower=0.3,
            split_by=1,
        )
        sim_data2 = hddm.generate.gen_rand_rlddm_data(
            a=a,
            t=t,
            scaler=scaler,
            alpha=alpha,
            pos_alpha=pos_alpha,
            size=size2,
            p_upper=0.6,
            p_lower=0.4,
            split_by=2,
        )
        # append the conditions
        sim_data0 = sim_data0.append([sim_data1, sim_data2], ignore_index=True)
        # assign subj_idx
        sim_data0["subj_idx"] = s
        # identify that these are simulated data
        sim_data0["type"] = "simulated"
        # identify the simulated data
        sim_data0["samp"] = i
        # append data from each subject
        sim_data = sim_data.append(sim_data0, ignore_index=True)
# combine observed and simulated data
ppc_dual_data = data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "samp"]
].copy()
ppc_dual_data["type"] = "observed"
ppc_dual_sdata = sim_data[
    ["subj_idx", "response", "split_by", "rt", "trial", "feedback", "type", "samp"]
].copy()
ppc_dual_data = ppc_dual_data.append(ppc_dual_sdata)
```

### Code cell 62

```python
# for practical reasons we only look at the first 40 trials for each subject in a given condition
plot_ppc_dual_data = ppc_dual_data[ppc_dual_data.trial < 41].copy()
```

### Choice

### Code cell 64

```python
# bin trials to for smoother estimate of response proportion across learning
plot_ppc_dual_data["bin_trial"] = pd.cut(
    plot_ppc_dual_data.trial, 11, labels=np.linspace(0, 10, 11)
).astype("int64")
# calculate means for each sample
sums = (
    plot_ppc_dual_data.groupby(["bin_trial", "split_by", "samp", "type"])
    .mean()
    .reset_index()
)
# calculate the overall mean response across samples
ppc_dual_sim = sums.groupby(["bin_trial", "split_by", "type"]).mean().reset_index()
# initiate columns that will have the upper and lower bound of the hpd
ppc_dual_sim["upper_hpd"] = 0
ppc_dual_sim["lower_hpd"] = 0
for i in range(0, ppc_dual_sim.shape[0]):
    # calculate the hpd/hdi of the predicted mean responses across bin_trials
    hdi = pymc.utils.hpd(
        sums.response[
            (sums["bin_trial"] == ppc_dual_sim.bin_trial[i])
            & (sums["split_by"] == ppc_dual_sim.split_by[i])
            & (sums["type"] == ppc_dual_sim.type[i])
        ],
        alpha=0.1,
    )
    ppc_dual_sim.loc[i, "upper_hpd"] = hdi[1]
    ppc_dual_sim.loc[i, "lower_hpd"] = hdi[0]
# calculate error term as the distance from upper bound to mean
ppc_dual_sim["up_err"] = ppc_dual_sim["upper_hpd"] - ppc_dual_sim["response"]
ppc_dual_sim["low_err"] = ppc_dual_sim["response"] - ppc_dual_sim["lower_hpd"]
ppc_dual_sim["model"] = "RLDDM_dual_learning"
```

### Code cell 65

```python
# plotting evolution of choice proportion for best option across learning for observed and simulated data.
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True)
for i in range(0, 3):
    ax = axs[i]
    d = ppc_dual_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "simulated")]
    ax.errorbar(
        d.bin_trial,
        d.response,
        yerr=[d.low_err, d.up_err],
        label="simulated",
        color="orange",
    )
    d = ppc_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "observed")]
    ax.plot(d.bin_trial, d.response, linewidth=3, label="observed")
    ax.set_title("split_by = %i" % i, fontsize=20)
    ax.set_ylabel("mean response")
    ax.set_xlabel("trial")
plt.legend()
```

__Fig.__ The plots display the rate of choosing the best option (response = 1) across learning and condition. The model generates data (orange) that closely follows the observed behavior (blue), with the exception of performance early in the most difficult condition (split_by=2).

### PPC for single vs. dual learning rate
To get a better sense of differences in ability to predict data between the single and dual learning rate model we can plot them together:

### Code cell 68

```python
# plotting evolution of choice proportion for best option across learning for observed and simulated data. Compared for model with single and dual learning rate.
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True)
for i in range(0, 3):
    ax = axs[i]
    d_single = ppc_sim[(ppc_sim.split_by == i) & (ppc_sim.type == "simulated")]
    # slightly move bin_trial to avoid overlap in errorbars
    d_single["bin_trial"] += 0.2
    ax.errorbar(
        d_single.bin_trial,
        d_single.response,
        yerr=[d_single.low_err, d_single.up_err],
        label="simulated_single",
        color="orange",
    )
    d_dual = ppc_dual_sim[
        (ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "simulated")
    ]
    ax.errorbar(
        d_dual.bin_trial,
        d_dual.response,
        yerr=[d_dual.low_err, d_dual.up_err],
        label="simulated_dual",
        color="green",
    )
    d = ppc_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "observed")]
    ax.plot(d.bin_trial, d.response, linewidth=3, label="observed")
    ax.set_title("split_by = %i" % i, fontsize=20)
    ax.set_ylabel("mean response")
    ax.set_xlabel("trial")
plt.xlim(-0.5, 10.5)
plt.legend()
```

__Fig.__ The predictions from the model with two learning rates are not very different from the model with single learning rate, and a similar overprediction of performance early on for the most difficult condition (split_by =2).

### RT

### Code cell 71

```python
plot_ppc_data["type_compare"] = np.where(
    plot_ppc_data["type"] == "observed",
    plot_ppc_data["type"],
    "simulated_single_learning",
)
plot_ppc_dual_data["type_compare"] = np.where(
    plot_ppc_dual_data["type"] == "observed",
    plot_ppc_dual_data["type"],
    "simulated_dual_learning",
)
dual_vs_single_pcc = plot_ppc_data.append(plot_ppc_dual_data)
dual_vs_single_pcc["reaction time"] = np.where(
    dual_vs_single_pcc["response"] == 1,
    dual_vs_single_pcc.rt,
    0 - dual_vs_single_pcc.rt,
)
# plotting evolution of choice proportion for best option across learning for observed and simulated data. We use bins of trials because plotting individual trials would be very noisy.
g = sns.FacetGrid(dual_vs_single_pcc, col="split_by", hue="type_compare", height=5)
g.map(sns.kdeplot, "reaction time", bw=0.01).set_ylabels("Density")
g.add_legend()
```

__Fig.__ Again there's not a big difference between the two models. Both models slightly overpredict performance for the medium (split_by =1) and hard (split_by = 2) conditions, as identified by lower densities for the negative (worst option choices) in the simulated compared to observed data.

### Transform alpha and pos_alpha
To interpret the parameter estimates for alpha and pos_alpha you have to transform them with the inverse logit where learning rate for negative prediction error is alpha and learning rate for positive prediction errors is pos_alpha. For this dataset the learning rate is estimated to be higher for positive than negative prediction errors.

### Code cell 74

```python
# plot alpha for positive and negative learning rate
traces = m_dual.get_traces()
neg_alpha = np.exp(traces["alpha"]) / (1 + np.exp(traces["alpha"]))
pos_alpha = np.exp(traces["pos_alpha"]) / (1 + np.exp(traces["pos_alpha"]))
sns.kdeplot(
    neg_alpha, color="r", label="neg_alpha: " + str(np.round(np.mean(neg_alpha), 3))
)
sns.kdeplot(
    pos_alpha, color="b", label="pos_alpha: " + str(np.round(np.mean(pos_alpha), 3))
)
```

__Fig.__ The positive learning rate is estimated to be stronger than the negative learning rate. Sticky choice, tendencies to repeat choices, could be driving some of this difference. The current model does not allow to test for this, however, but it could be tested in the future if we implement a regression version of RLDDM (similar to HDDMRegressor).

### Simulate data with learning rates for positive and negative prediction errors
Here's how you would simulate data with a learning rate for positive and negative predictions of 0.2 and 0.4, respectively:

### Code cell 77

```python
hddm.generate.gen_rand_rlddm_data(
    a=1, t=0.3, alpha=0.2, pos_alpha=0.4, scaler=2, p_upper=0.8, p_lower=0.2, size=10
)
```

## 10. depends_on vs. split_by
HDDMrl can be used to estimate separate parameters just as in the standard HDDM. But in RL you typically estimate the same learning rates and inverse temperature across conditions. That's one reason why you have to specify condition in the split_by-column instead of depends_on. (The other is that if you use depends_on the expected rewards will not get updated properly). But depends_on is still useful, for example if you want to estimate the effect of group on parameters. As an example we can simulate a dataset with two groups that have different decision thresholds:

### Code cell 79

```python
data1 = hddm.generate.gen_rand_rlddm_data(
    a=1, t=0.3, alpha=0.4, scaler=2, p_upper=0.8, p_lower=0.2, subjs=50, size=50
)
data1["group"] = "group1"
data2 = hddm.generate.gen_rand_rlddm_data(
    a=2, t=0.3, alpha=0.4, scaler=2, p_upper=0.8, p_lower=0.2, subjs=50, size=50
)
data2["group"] = "group2"
group_data = data1.append(data2)
group_data["q_init"] = 0.5
m = hddm.HDDMrl(
    group_data, depends_on={"v": "group", "a": "group", "t": "group", "alpha": "group"}
)
m.sample(1500, burn=500, dbname="traces.db", db="pickle")
```

### Code cell 80

```python
# the plot shows that the model was able to recover the different decision threshold across groups.
a_group1, a_group2 = m.nodes_db.node[["a(group1)", "a(group2)"]]
hddm.analyze.plot_posterior_nodes([a_group1, a_group2])
plt.xlabel("decision threshold")
plt.ylabel("Posterior probability")
plt.xlim(0.7, 2.3)
plt.title("Posterior of decision threshold group means")
```

## 11. Probabilistic binary outcomes vs. normally distributed outcomes
The examples so far have all been using a task structure where the outcomes are binary and probabilistic. But the model can also be applied to other types of outcomes. Here we show how you can generate and model data with normally distributed outcomes. As you will see you don't have to do any modifications to the model estimation process, but you have to change the input for generating data. Also note that the scaling parameter (v) will scale negatively with the values of the observed outcomes because the combined drift rate needs to be plausible.

### Code cell 82

```python
# This is how we generated data so far, defining the probability of reward (1) for actions/stimuli associated with upper and lower boundary.
# binary probabilistic outcomes
hddm.generate.gen_rand_rlddm_data(
    a=2, t=0.3, scaler=2, alpha=0.2, size=10, p_upper=0.2, p_lower=0.8
)
```

### Code cell 83

```python
# If instead the outcomes are drawn from a normal distribution you will have to set binary_outcome to False and instead of p_upper and p_upper define the mean (mu) and sd
# of the normal distribution for both alternatives. Note that we change the initial q-value to 0, and that we reduce the scaling factor.
# normally distributed outcomes
hddm.generate.gen_rand_rlddm_data(
    a=2,
    t=0.3,
    scaler=0.2,
    alpha=0.2,
    size=10,
    mu_upper=8,
    mu_lower=2,
    sd_upper=1,
    sd_lower=1,
    binary_outcome=False,
    q_init=0,
)
```

### Code cell 84

```python
# We can generate a dataset where 30 subjects perform 50 trials each. Note that we set the scaler to be lower than for the binary outcomes as otherwise
# the resulting drift will be unrealistically high.
norm_data = hddm.generate.gen_rand_rlddm_data(
    a=2,
    t=0.3,
    scaler=0.2,
    alpha=0.2,
    size=50,
    subjs=30,
    mu_upper=8,
    mu_lower=2,
    sd_upper=2,
    sd_lower=2,
    binary_outcome=False,
    q_init=0,
)
```

### Code cell 85

```python
# and then we can do estimation as usual
# but first we need to define inital q-value
norm_data["q_init"] = 0
m_norm = hddm.HDDMrl(norm_data)
m_norm.sample(1500, burn=500, dbname="traces.db", db="pickle")
m_norm.print_stats()
```

## 12. HDDMrlRegressor

As of version 0.7.6. HDDM includes a module for estimating the impact of continuous regressor onto RLDDM parameters. The module, called HDDMrlRegressor, works the same way as the HDDMRegressor for the normal DDM. The method allows estimation of the association of e.g. neural measures onto parameters. To illustrate the method we extend the function to generate rlddm_data by adding a normally distributed regressor and including a coefficient called 'neural'.Note that to run the HDDMrlRegressor you need to include alpha when specifying the model. For more information on how to set up regressor models look at the tutorial for HDDM.

### Code cell 87

```python
# function to generate rlddm-data that adds a neural regressor to decision threshold
def gen_rand_reg_rlddm_data(
    a,
    t,
    scaler,
    alpha,
    neural,
    size=1,
    p_upper=1,
    p_lower=0,
    z=0.5,
    q_init=0.5,
    split_by=0,
    subjs=1,
):
    all_data = []
    n = size
    # set sd for variables to generate subject-parameters from group distribution
    sd_t = 0.02
    sd_a = 0.1
    sd_alpha = 0.1
    sd_v = 0.25
    # save parameter values as group-values
    tg = t
    ag = a
    alphag = alpha
    scalerg = scaler
    for s in range(0, subjs):
        t = (
            np.maximum(0.05, np.random.normal(loc=tg, scale=sd_t, size=1))
            if subjs > 1
            else tg
        )
        a = (
            np.maximum(0.05, np.random.normal(loc=ag, scale=sd_a, size=1))
            if subjs > 1
            else ag
        )
        alpha = (
            np.minimum(
                np.minimum(
                    np.maximum(0.001, np.random.normal(loc=alphag, scale=sd_a, size=1)),
                    alphag + alphag,
                ),
                1,
            )
            if subjs > 1
            else alphag
        )
        scaler = (
            np.random.normal(loc=scalerg, scale=sd_v, size=1) if subjs > 1 else scalerg
        )
        # create a normalized regressor that is combined with the neural coefficient to create trial-by-trial values for decision threshold
        neural_reg = np.random.normal(0, 1, size=n)
        q_up = np.tile([q_init], n)
        q_low = np.tile([q_init], n)
        response = np.tile([0.5], n)
        feedback = np.tile([0.5], n)
        rt = np.tile([0], n)
        rew_up = np.random.binomial(1, p_upper, n).astype(float)
        rew_low = np.random.binomial(1, p_lower, n).astype(float)
        sim_drift = np.tile([0], n)
        subj_idx = np.tile([s], n)
        d = {
            "q_up": q_up,
            "q_low": q_low,
            "sim_drift": sim_drift,
            "rew_up": rew_up,
            "rew_low": rew_low,
            "response": response,
            "rt": rt,
            "feedback": feedback,
            "subj_idx": subj_idx,
            "split_by": split_by,
            "trial": 1,
            "neural_reg": neural_reg,
        }
        df = pd.DataFrame(data=d)
        df = df[
            [
                "q_up",
                "q_low",
                "sim_drift",
                "rew_up",
                "rew_low",
                "response",
                "rt",
                "feedback",
                "subj_idx",
                "split_by",
                "trial",
                "neural_reg",
            ]
        ]
        # generate data trial-by-trial using the Intercept (a), regressor (neural_reg) and coefficient (neural) for decision threshold.
        data, params = hddm.generate.gen_rand_data(
            {
                "a": a + neural * df.loc[0, "neural_reg"],
                "t": t,
                "v": df.loc[0, "sim_drift"],
                "z": z,
            },
            subjs=1,
            size=1,
        )
        df.loc[0, "response"] = data.response[0]
        df.loc[0, "rt"] = data.rt[0]
        if data.response[0] == 1.0:
            df.loc[0, "feedback"] = df.loc[0, "rew_up"]
        else:
            df.loc[0, "feedback"] = df.loc[0, "rew_low"]

        for i in range(1, n):
            df.loc[i, "trial"] = i + 1
            df.loc[i, "q_up"] = (
                df.loc[i - 1, "q_up"] * (1 - df.loc[i - 1, "response"])
            ) + (
                (df.loc[i - 1, "response"])
                * (
                    df.loc[i - 1, "q_up"]
                    + (alpha * (df.loc[i - 1, "rew_up"] - df.loc[i - 1, "q_up"]))
                )
            )
            df.loc[i, "q_low"] = (
                df.loc[i - 1, "q_low"] * (df.loc[i - 1, "response"])
            ) + (
                (1 - df.loc[i - 1, "response"])
                * (
                    df.loc[i - 1, "q_low"]
                    + (alpha * (df.loc[i - 1, "rew_low"] - df.loc[i - 1, "q_low"]))
                )
            )
            df.loc[i, "sim_drift"] = (df.loc[i, "q_up"] - df.loc[i, "q_low"]) * (scaler)
            data, params = hddm.generate.gen_rand_data(
                {
                    "a": a + neural * df.loc[i, "neural_reg"],
                    "t": t,
                    "v": df.loc[i, "sim_drift"],
                    "z": z,
                },
                subjs=1,
                size=1,
            )
            df.loc[i, "response"] = data.response[0]
            df.loc[i, "rt"] = data.rt[0]
            if data.response[0] == 1.0:
                df.loc[i, "feedback"] = df.loc[i, "rew_up"]
            else:
                df.loc[i, "feedback"] = df.loc[i, "rew_low"]
        all_data.append(df)
    all_data = pd.concat(all_data, axis=0)
    all_data = all_data[
        [
            "q_up",
            "q_low",
            "sim_drift",
            "response",
            "rt",
            "feedback",
            "subj_idx",
            "split_by",
            "trial",
            "neural_reg",
        ]
    ]

    return all_data
```

### Code cell 88

```python
# Create data with function defined above.
# This will create trial-by-trial values for decision threshold (a) by adding the coefficient neural (here set to 0.2)
# multiplied by a normalized regressor (neural_reg) to the 'Intercept' value of a (here set to 1)
data_neural = gen_rand_reg_rlddm_data(
    a=1,
    t=0.3,
    scaler=2,
    alpha=0.2,
    neural=0.2,
    size=100,
    p_upper=0.7,
    p_lower=0.3,
    subjs=25,
)
data_neural["q_init"] = 0.5
data_neural.head()
```

### Code cell 89

```python
# run a regressor model estimating the impact of 'neural' on decision threshold a. This should estimate the coefficient a_neural_reg to be 0.2
# to run the HDDMrlRegressor you need to include alpha
m_reg = hddm.HDDMrlRegressor(data_neural, "a ~ neural_reg", include=["v", "a", "t", "alpha")
m_reg.sample(1000, burn=250)
m_reg.print_stats()
```

## 13. Regular RL without RT
HDDMrl also includes a module to run an RL-model that uses softmax to transform q-values to probability of choosing options associated with upper (response=1) or lower (response=0) boundary. To run this model you type hddm.Hrl instead of hddm.HDDMrl. The setup is the same as for HDDMrl, and for now, the model won't run if you don't include an rt-column. This will be fixed for a future version, but for now, if you don't have RTs you can just create an rt-column where you set all rts to e.g. 0.5. You can choose to estimate separate learning rates for positive and negative learning rate by setting dual=True (see [here](#9.-Separate-learning-rates-for-positive-and-negative-prediction-errors) for more information). The model will by default estimate posterior distributions for the alpha and v parameters. The probability of choosing upper boundary is captured as: <br><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$p_{up} =(e^{-2*z*d_t}-1)/ (e^{-2*d_t}-1)$, <br><br>
where ${d_t}=q_{up_t}-q_{low}*v$ and z represents starting point (which for now is fixed to be 0.5). <br>
This calculation is equivalent to soft-max transformation when z=0.5.

### Code cell 91

```python
# run the model by calling hddm.Hrl (instead of hddm.HDDM for normal model and hddm.HDDMrl for rlddm-model)
m_rl = hddm.Hrl(data)
# set sample and burn-in
m_rl.sample(1500, burn=500, dbname="traces.db", db="pickle")
# print stats to get an overview of posterior distribution of estimated parameters
m_rl.print_stats()
```

Parameter estimates from the pure RL-model are a bit different compared to the RLDDM. This is to be expected as probability of choice in DDM is dependent both on the decsision threshold and the scaled difference in q-values, whereas the RL model only uses the scaled difference in q-values.

### Code cell 93

```python
m_rl.plot_posteriors()
```

__Fig.__ Mixing and autocorrelation looks good.

### Code cell 95

```python
# estimate convergence
models = []
for i in range(3):
    m = hddm.Hrl(data=data)
    m.sample(1500, burn=500, dbname="traces.db", db="pickle")
    models.append(m)
# get max gelman-statistic value. shouldn't be higher than 1.1
np.max(list(gelman_rubin(models).values()))
```

Convergence looks good, i.e. no parameters with gelman-rubin statistic > 1.1.

### Code cell 97

```python
# Create a new model that has all traces concatenated
# of individual models.
m_rl = kabuki.utils.concat_models(models)
```

### Code cell 98

```python
alpha, v = m_rl.nodes_db.node[["alpha", "v"]]
samples = {"alpha": alpha.trace(), "v": v.trace()}
samp = pd.DataFrame(data=samples)


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), xy=(0.1, 0.9), xycoords=ax.transAxes)


g = sns.PairGrid(samp, palette=["red"])
g.map_upper(plt.scatter, s=10)
g.map_diag(sns.distplot, kde=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_lower(corrfunc)
```

__Fig.__ The correlation in the posterior distribution for alpha and v/scaling is somewhat negative.

## Posterior predictive check
We can also do posterior predictive check on the RL-model by generating new data with hddm.generate.gen_rand_rl_data.

### Code cell 101

```python
# create empty dataframe to store simulated data
sim_data = pd.DataFrame()
# create a column samp to be used to identify the simulated data sets
data["samp"] = 0
# load traces
traces = m_rl.get_traces()
# decide how many times to repeat simulation process. repeating this multiple times is generally recommended as it better captures the uncertainty in the posterior distribution, but will also take some time
for i in tqdm(range(1, 51)):
    # randomly select a row in the traces to use for extracting parameter values
    sample = np.random.randint(0, traces.shape[0] - 1)
    # loop through all subjects in observed data
    for s in data.subj_idx.unique():
        # get number of trials for each condition.
        size0 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 0)].trial.unique()
        )
        size1 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 1)].trial.unique()
        )
        size2 = len(
            data[(data["subj_idx"] == s) & (data["split_by"] == 2)].trial.unique()
        )
        # set parameter values for simulation
        scaler = traces.loc[sample, "v_subj." + str(s)]
        alphaInv = traces.loc[sample, "alpha_subj." + str(s)]
        # take inverse logit of estimated alpha
        alpha = np.exp(alphaInv) / (1 + np.exp(alphaInv))
        # simulate data for each condition changing only values of size, p_upper, p_lower and split_by between conditions.
        sim_data0 = hddm.generate.gen_rand_rl_data(
            scaler=scaler, alpha=alpha, size=size0, p_upper=0.8, p_lower=0.2, split_by=0
        )
        sim_data1 = hddm.generate.gen_rand_rl_data(
            scaler=scaler, alpha=alpha, size=size1, p_upper=0.7, p_lower=0.3, split_by=1
        )
        sim_data2 = hddm.generate.gen_rand_rl_data(
            scaler=scaler, alpha=alpha, size=size2, p_upper=0.6, p_lower=0.4, split_by=2
        )
        # append the conditions
        sim_data0 = sim_data0.append([sim_data1, sim_data2], ignore_index=True)
        # assign subj_idx
        sim_data0["subj_idx"] = s
        # identify that these are simulated data
        sim_data0["type"] = "simulated"
        # identify the simulated data
        sim_data0["samp"] = i
        # append data from each subject
        sim_data = sim_data.append(sim_data0, ignore_index=True)
# combine observed and simulated data
ppc_rl_data = data[
    ["subj_idx", "response", "split_by", "trial", "feedback", "samp"]
].copy()
ppc_rl_data["type"] = "observed"
ppc_rl_sdata = sim_data[
    ["subj_idx", "response", "split_by", "trial", "feedback", "type", "samp"]
].copy()
ppc_rl_data = ppc_rl_data.append(ppc_rl_sdata)
```

### Code cell 102

```python
# for practical reasons we only look at the first 40 trials for each subject in a given condition
plot_ppc_rl_data = ppc_rl_data[ppc_rl_data.trial < 41].copy()
```

### Code cell 103

```python
# bin trials to for smoother estimate of response proportion across learning
plot_ppc_rl_data["bin_trial"] = pd.cut(
    plot_ppc_rl_data.trial, 11, labels=np.linspace(0, 10, 11)
).astype("int64")
# calculate means for each sample
sums = (
    plot_ppc_rl_data.groupby(["bin_trial", "split_by", "samp", "type"])
    .mean()
    .reset_index()
)
# calculate the overall mean response across samples
ppc_rl_sim = sums.groupby(["bin_trial", "split_by", "type"]).mean().reset_index()
# initiate columns that will have the upper and lower bound of the hpd
ppc_rl_sim["upper_hpd"] = 0
ppc_rl_sim["lower_hpd"] = 0
for i in range(0, ppc_rl_sim.shape[0]):
    # calculate the hpd/hdi of the predicted mean responses across bin_trials
    hdi = pymc.utils.hpd(
        sums.response[
            (sums["bin_trial"] == ppc_rl_sim.bin_trial[i])
            & (sums["split_by"] == ppc_rl_sim.split_by[i])
            & (sums["type"] == ppc_rl_sim.type[i])
        ],
        alpha=0.1,
    )
    ppc_rl_sim.loc[i, "upper_hpd"] = hdi[1]
    ppc_rl_sim.loc[i, "lower_hpd"] = hdi[0]
# calculate error term as the distance from upper bound to mean
ppc_rl_sim["up_err"] = ppc_rl_sim["upper_hpd"] - ppc_rl_sim["response"]
ppc_rl_sim["low_err"] = ppc_rl_sim["response"] - ppc_rl_sim["lower_hpd"]
ppc_rl_sim["model"] = "RL"
```

### Code cell 104

```python
# plotting evolution of choice proportion for best option across learning for observed and simulated data. Compared for RL and RLDDM models, both with single learnign rate.
fig, axs = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharex=True, sharey=True)
for i in range(0, 3):
    ax = axs[i]
    d_single = ppc_sim[(ppc_sim.split_by == i) & (ppc_sim.type == "simulated")]
    # slightly move bin_trial to avoid overlap in errorbars
    d_single["bin_trial"] += 0.2
    ax.errorbar(
        d_single.bin_trial,
        d_single.response,
        yerr=[d_single.low_err, d_single.up_err],
        label="simulated_RLDDM",
        color="orange",
    )
    ax = axs[i]
    d_rl = ppc_rl_sim[(ppc_rl_sim.split_by == i) & (ppc_rl_sim.type == "simulated")]
    ax.errorbar(
        d_rl.bin_trial,
        d_rl.response,
        yerr=[d_rl.low_err, d_rl.up_err],
        label="simulated_RL",
        color="green",
    )
    ax = axs[i]
    d = ppc_sim[(ppc_dual_sim.split_by == i) & (ppc_dual_sim.type == "observed")]
    ax.plot(d.bin_trial, d.response, linewidth=3, label="observed")
    ax.set_title("split_by = %i" % i, fontsize=20)
    ax.set_ylabel("mean response")
    ax.set_xlabel("trial")
plt.xlim(-0.5, 10.5)
plt.legend()
```

__Fig.__ The predicted choice for the RL-model is very similar to what was predicted in the RLDDM. That is not surprising given that they use the same calculation to get the choice likelihood. The difference between them is instead that the RLDDM could potentially detect the unique contribution of the scaling/drift parameter and the decision threshold onto choice.

### Misprediction across learning
Another way to visualize this is to look at how the predicted choice misses on the observed across learning, i.e. predicted-observed. As for the other plots we see that the two methods are very similar.

### Code cell 107

```python
# rl
error_prediction = (
    plot_ppc_rl_data.groupby(["split_by", "type", "bin_trial"])["response"]
    .mean()
    .reset_index()
)
ep = error_prediction.pivot_table(
    index=["split_by", "bin_trial"], columns="type", values="response"
).reset_index()
ep["diff"] = ep["simulated"] - ep["observed"]
ep["model"] = "RL"
# rlddm
error_prediction = (
    plot_ppc_data.groupby(["split_by", "type", "bin_trial"])["response"]
    .mean()
    .reset_index()
)
ep_rlddm = error_prediction.pivot_table(
    index=["split_by", "bin_trial"], columns="type", values="response"
).reset_index()
ep_rlddm["diff"] = ep_rlddm["simulated"] - ep_rlddm["observed"]
ep_rlddm["model"] = "RLDDM"
# combine
ep = ep.append(ep_rlddm)
# plot
g = sns.relplot(
    x="bin_trial",
    y="diff",
    col="split_by",
    hue="model",
    kind="line",
    ci=False,
    data=ep,
    palette="Set2_r",
)
g.map(plt.axhline, y=0, ls=":", c=".5")
```
