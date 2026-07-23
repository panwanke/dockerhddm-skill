# Source map and precedence

## Precedence

1. The commits pinned as submodules by `{dockerhddm_repo}` define dockerHDDM
   1.1.0 behavior.
2. Tests and notebooks at the same repository revision document intended use.
3. Separate custom clones record earlier maintenance history.
4. Articles and old tutorials are explanatory; confirm their code against 1–3.

## Pinned v1.1 source

| Component | Pinned commit | Relevant change |
|---|---|---|
| HDDM | `4205656814327fb709421e3f2287ba13c58e00e2` | NumPy 2/Cython 3 compatibility; follows stimcoding fix `b6e6c3f` |
| Kabuki | `a2302364b8b76c36c10e386520cea6917d6c0565` | DataTree plotting; follows DataTree `to_infdata` refactor |
| PyMC2 | `ed2dc4ca0f2c2bf48fa2f07909a1c36f8bac1b2f` | NumPy 2 support; modern setuptools/F2PY build |
| ssm-simulators | `9eaaad72fda953d17b726de3272edfcac5cdf5cd` | Stable dockerHDDM branch and NumPy 2 build |

Resolve the repository from `paths.dockerhddm_repo` in `env.json`.

## Legacy custom clones

The configured clones are older than the pinned submodules:

- `paths.hddm_custom_legacy`: commit `5e4dac1`;
- `paths.kabuki_custom_legacy`: commit `4be59f0`;
- `paths.pymc2_custom_legacy`: commit `8c36a4fa` on `dev`.

Use them to understand historical changes, reproduce old failures, or compare
behavior. Do not use them to answer “what does dockerHDDM 1.1.0 run?” without
checking the pinned submodules.

## High-value customized files

| Concern | File |
|---|---|
| multi-chain sampling, saving, InferenceData | `kabuki/kabuki/hierarchical.py` |
| PPC and pointwise log-likelihood parallelism | `kabuki/kabuki/analyze.py` |
| DataTree plotting | `kabuki/kabuki/analyze.py` |
| classic simulation | `hddm/hddm/generate.py` |
| hierarchical/LAN simulation | `hddm/hddm/simulators/hddm_dataset_generators.py` |
| model parameters and bounds | `hddm/hddm/model_config.py` |
| stimulus coding | `hddm/hddm/models/hddm_stimcoding.py` |
| PyMC2 sampler signature | `pymc2/pymc/MCMC.py` |
| reproducible build | `Dockerfile`, submodule `setup.py`/`pyproject.toml` |

## Re-check commands

```bash
git -C <dockerhddm-repo> submodule status
git -C <dockerhddm-repo>/kabuki log -8 --oneline
git -C <dockerhddm-repo>/hddm log -8 --oneline
git -C <dockerhddm-repo>/pymc2 log -8 --oneline
```
