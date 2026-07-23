# Fixed project layout

```text
project/
├── README.md
├── CHANGELOG.md
├── config/
│   └── model.json
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── common/
│   ├── inference/
│   └── simulation/
├── notebooks/
├── models/
├── results/
│   ├── diagnostics/
│   ├── ppc/
│   └── simulation/
├── figs/
├── docs/
│   └── data-dictionary.md
└── logs/
```

## Roles

- `data/raw`: immutable source data; normally untracked.
- `data/processed`: validated HDDM-ready tables.
- `config`: declarative model/sampling/simulation choices.
- `scripts/common`: validation and shared utilities.
- `scripts/inference`: real-data fitting, diagnostics, PPC, comparison.
- `scripts/simulation`: generation, recovery, design sensitivity, plots.
- `notebooks`: exploration only; move stable logic into scripts.
- `models`: `.db`, `.hddm`, `.nc`; large and ignored.
- `results`: machine-readable tables/metrics.
- `figs`: rendered outputs, never primary data.
- `docs`: coding, assumptions, analysis decisions.
- `logs`: run metadata and captured output.

## Numbered workflow

Use numeric prefixes within each workflow:

- `00_validate_data.py`
- `10_fit.py` / `10_generate.py`
- `20_diagnostics.py` / `20_parameter_recovery.py`
- `30_ppc.py` / `30_visualize.py`
- `40_compare.py`
- `50_report.py`

## Separation rules

- Never mix raw-data transformation with plotting.
- Never write generated results beside source code.
- Never overwrite raw data.
- Store ground truth and seeds with simulations.
- Store runtime/image/source pins with fits.
- Update README for usage and CHANGELOG for durable changes.
- Use generic project structure only; do not copy sensitive identifiers or
  unpublished labels from reference projects.
