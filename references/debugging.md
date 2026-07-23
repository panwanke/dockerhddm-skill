# Debugging ladder

## 1. Environment

Run `scripts/inspect_environment.py`. Check image tag, component versions,
submodule pins, CPU architecture, import order, and whether the failure occurs
inside the official container. “PyMC” documentation found online usually
describes modern PyMC, not PyMC2.

Build/import failures on Python 3.12 or NumPy 2 usually indicate an unpinned
legacy clone, stale wheel/cache, or installation outside dockerHDDM 1.1.

## 2. Mount and persistence

Confirm the host project is mounted at `/home/jovyan/work`, is writable, and
artifacts appear on the host. With `--rm`, files outside the mount disappear.
Use unique `save_name`/database paths for concurrent fits.

## 3. Data

Check required columns, nulls, duplicated trials, positive RT, seconds vs.
milliseconds, binary response values, response meaning, stimulus levels,
condition balance, per-subject counts, error rates, outliers, and trial-order
trends. A syntactically valid model can be scientifically invalid because
coding is reversed or errors are too rare.

## 4. Model construction

Start with `HDDM(data, include=["v","a","t"])`. Add `z`, variability,
condition dependencies, or regression terms one at a time. Do not use
`depends_on` and regression for the same parameter. Verify all subjects contain
the design cells needed by a regression matrix.

For stimulus coding, use the pinned submodule containing commit `b6e6c3f`;
older clones may replace or mishandle the include list.

## 5. Sampling

- smoke-test 100–500 iterations;
- give every concurrent chain/model a unique save path;
- verify the number of surviving chains;
- inspect printed per-chain errors;
- do not treat a saved file as evidence of convergence;
- reduce parallelism when memory or serialization is unstable.

## 6. InferenceData, PPC, and log-likelihood

Generate groups after sampling. Do not pass `n_jobs` to `sample()` in the
pinned source. Assert the groups you requested because fallback conversion can
return an object without them.

For DataTree/ArviZ failures, inspect capabilities rather than forcing an old
API. Use `az.plot_ppc_dist` or pinned Kabuki plotting helpers.

## 7. Memory/OOM

Estimate arrays:

```bash
python scripts/estimate_memory.py \
  --subjects 100 --trials-per-subject 1000 \
  --parameters 500 --draws 2000 --chains 4 --n-ppc 100
```

Major costs:

- trace: parameter nodes × draws × chains;
- PPC: observations × PPC draws × chains × output variables;
- pointwise log-likelihood: observations × retained draws × chains;
- merge peak: Joblib results, model deep copies, pandas/xarray conversions.

Mitigations:

1. sample and convert in separate stages;
2. set `n_jobs=1` or `2` in `to_infdata`;
3. bound `n_ppc` (often 50–200 for diagnostic plots);
4. avoid `loglike=True` until required;
5. use subsampled PSIS-LOO for large comparisons, with shared observation
   indices and Pareto-k diagnostics;
6. save artifacts to disk and restart between heavy stages;
7. increase WSL memory/swap only after controlling software multipliers.

## 8. Statistical/modeling failure

If chains run but results are unstable, inspect R-hat/ESS, posterior geometry,
prior sensitivity, parameter correlations, recovery, and individual PPC.
Simplify unidentifiable variability parameters, gather more informative data,
or redesign the task.

## 9. Theory/design failure

Investigate ceiling accuracy, fewer than ~10 errors in a cell, nonstationarity,
truncated RT distributions, changing evidence within a trial, feedback-driven
learning, response deadlines, and contaminant strategies. These cannot be
fixed by more MCMC draws.
