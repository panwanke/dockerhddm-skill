# Customized Kabuki and PyMC2 API

## Call flow

```text
hddm.HDDM / HDDMRegressor
  → kabuki.Hierarchical builds Knode/PyMC2 node graph
  → Hierarchical.sample orchestrates chain copies
  → pymc.MCMC.sample runs each chain
  → concat_models merges traces
  → Hierarchical.to_infdata builds xarray/ArviZ groups
  → kabuki.analyze generates PPC and pointwise log-likelihood
```

## Customized `sample`

Pinned file: `kabuki/kabuki/hierarchical.py`.

Custom keywords include `chains`, `parallel`, `return_infdata`, `save_name`,
`sample_prior`, `n_prior`, `loglike`, `n_loglike`, `ppc`, `n_ppc`, and
`find_starting_values`. More than one chain deep-copies the HDDM object and may
use Joblib. Failed chains are caught and removed; verify the returned chain
count instead of assuming all requested chains survived.

When `save_name` or InferenceData is requested, the wrapper uses a disk-backed
pickle database and creates parent directories.

## Critical keyword boundary

The pinned wrapper pops its custom keywords, then forwards remaining `kwargs`
to `pymc.MCMC.sample`. PyMC2 accepts a fixed signature and does not accept
`n_jobs`. Therefore:

```python
# Unsafe for the pinned source:
model.sample(2000, n_jobs=1, ppc=True)

# Safe staged form:
model.sample(
    2000, burn=1000, chains=4,
    return_infdata=True, save_name="models/m0"
)
model.to_infdata(
    ppc=True, n_ppc=100,
    loglike=False,
    parallel=True, n_jobs=1,
    save_name="models/m0",
)
```

`n_jobs` belongs to Kabuki analysis functions invoked during
`to_infdata`, not the PyMC2 sampler.

## Error masking

If automatic `to_infdata` fails inside `sample`, the wrapper prints an error and
tries a minimal `self.to_infdata()` fallback. Consequently, a returned object
may lack requested `posterior_predictive` or `log_likelihood`. Assert the
required groups after sampling.

## DataTree migration

Kabuki commits `9211711`, `dfc5708`, and `a230236` adapt ArviZ 1.1/xarray
DataTree behavior. Old recipes using legacy `InferenceData.add_groups()` or
old `az.plot_ppc` assumptions may fail. Use runtime feature checks and the
converted v1.1 notebooks.

## PyMC2 maintenance

The pinned PyMC2 is version 2.3.8 with a modernized build:

- setuptools/F2PY replaces removed `numpy.distutils`;
- deprecated `imp` imports are removed;
- NumPy 2 data-type/C-API compatibility is added;
- Python 3.12 compilation is supported.

This is not modern PyMC (v4/v5). Do not apply modern `pymc.sample` or NUTS
documentation to HDDM's PyMC2 node graph.
