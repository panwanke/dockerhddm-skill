# {{PROJECT_NAME}}

Workflow mode: **{{WORKFLOW_MODE}}**. Initialized on {{DATE}}.

## Runtime

Use dockerHDDM 1.1.0. Mount this project at `/home/jovyan/work`:

```bash
docker run -it --rm \
  -v "$(pwd):/home/jovyan/work" \
  -p 8888:8888 \
  hcp4715/hddm:1.1.0 \
  jupyter notebook
```

## Data contract

Trial-level input belongs in `data/raw/`. Never edit it in place. Write the
validated modeling table to `data/processed/trials.csv`. Required HDDM columns:

- `subj_idx`: stable subject identifier;
- `rt`: reaction time in seconds;
- `response`: binary boundary/choice coding documented in `docs/data-dictionary.md`;
- every column named in `depends_on` or a regression formula.

## Reproducible order

1. Record data definitions in `docs/data-dictionary.md`.
2. Run `python scripts/common/00_validate_data.py`.
3. For inference, edit `config/model.json`, then run scripts in
   `scripts/inference/` by numeric prefix.
4. For simulation, record generating parameters, then run scripts in
   `scripts/simulation/` by numeric prefix.
5. Save generated tables to `results/`, figures to `figs/`, and model artifacts
   to `models/`. Do not commit large `.db`, `.hddm`, or `.nc` files.
6. Record durable changes in `CHANGELOG.md`.

Smoke-test with small draws/trials first. Scale only after data, model
construction, serialization, and plots all pass.
