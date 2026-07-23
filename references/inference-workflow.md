# Inference workflow: data → parameters

## 1. Define the claim before the model

Write the intended contrast, its parameter mapping, the response coding, and
what observation would contradict the claim. Do not vary every DDM parameter
merely because the API permits it.

## 2. Validate trial-level data

Require `subj_idx`, positive `rt` in seconds, and binary `response`. Add
condition/covariate columns used by `depends_on` or Patsy formulas. Report:

- subject and trial counts;
- missing/nonresponse counts;
- error/choice rate by subject and condition;
- RT 0.1/0.5/0.9 quantiles by response and condition;
- fast/slow outliers and time-on-task trends.

Document whether response coding is accuracy, physical choice, or stimulus
coding. Never use `flip_errors` without explaining why that likelihood/coding
requires signed RT.

## 3. Choose the model surface

| Need | Class/API |
|---|---|
| standard hierarchical DDM | `hddm.HDDM(data, include=...)` |
| cell-wise condition variation | `depends_on={"v": "condition"}` |
| trial covariate/regression | `hddm.HDDMRegressor(data, "v ~ x")` |
| physical stimulus/choice bias | `hddm.HDDMStimCoding(...)` |
| nonstandard LAN model | `hddm.HDDMnn(..., model="...")` |

Do not use both a regression and `depends_on` for the same parameter.
`keep_regressor_trace=True` is needed for some regression PPC operations but
substantially increases storage.

## 4. Fit baseline before complexity

```python
baseline = hddm.HDDM(
    data,
    include=["v", "a", "t", "z"],
    p_outlier=0.05,
)
idata = baseline.sample(
    2500,
    burn=1000,
    chains=4,
    parallel=True,
    return_infdata=True,
    save_name="models/baseline",
)
```

Smoke-test first with ~100–500 iterations. Production values depend on mixing,
effective sample size, and recovery—not a universal number.

## 5. Add expensive groups separately

```python
idata = baseline.to_infdata(
    ppc=True,
    n_ppc=100,
    loglike=False,
    parallel=True,
    n_jobs=1,
    save_name="models/baseline",
)
```

Generate full pointwise log-likelihood only when model comparison requires it.
Use `n_loglike` for an explicit reduced diagnostic if scientifically
appropriate. Do not pass `n_jobs` to `sample()` in the pinned source.

## 6. Diagnose sampling

Inspect `az.summary`, trace/rank plots, R-hat, bulk/tail ESS, divergences or
step-method warnings, and between-chain agreement. A parameter can have a
plausible mean while its chain is not trustworthy.

## 7. Assess absolute fit

PPC must cover both choices/errors and the RT distribution, including at least
0.1, 0.5, and 0.9 quantiles per relevant design cell. Inspect individuals when
possible; a group-average PPC can hide systematic misfit.

## 8. Compare models

Use models fit to the same observations and coding. For large pointwise
log-likelihood arrays, consider ArviZ 1.1 `loo_subsample`; reuse identical
observation indices across models and still inspect Pareto-k.

Relative fit does not establish absolute adequacy. Interpret parameters only
after convergence, recovery, and PPC.

## 9. Report

Report data exclusions, units/coding, priors, varied/fixed parameters, sampling
settings, convergence, absolute fit, model comparison, recovery, parameter
summaries, software/image version, seed, and artifact paths.
