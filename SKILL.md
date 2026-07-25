---
name: dockerhddm-skill
description: >
  Build, explain, simulate, infer, organize, and debug dockerHDDM 1.1 projects
  using the maintained HDDM, Kabuki, PyMC2, ArviZ, and ssm-simulators stack.
  Use when the user explicitly invokes /dockerhddm-workflow or asks for a
  dockerHDDM/HDDM project, trial-level parameter inference, forward simulation,
  synthetic prediction, parameter recovery, PPC/LOO, API interpretation,
  memory/kernel debugging, model-theory troubleshooting, or EAM task-design
  guidance.
---

# dockerHDDM skill

Version: **0.2.5**  
Slash command: **`/dockerhddm-workflow`**

## Route the request

Classify before writing code:

| Intent | Evidence available | Primary route |
|---|---|---|
| Inference | Trial-level data exist | Read [inference-workflow.md](references/inference-workflow.md) |
| Simulation/prediction | Parameters, ranges, or theory exist | Read [simulation-prediction-workflow.md](references/simulation-prediction-workflow.md) |
| Debugging | Error, crash, implausible result, or design concern | Read [debugging.md](references/debugging.md) |
| Project initialization | User wants a clean reusable project | Run `scripts/init_project.py` and read [project-layout.md](references/project-layout.md) |
| API explanation | User asks how HDDM/Kabuki/PyMC2 calls flow | Read [api-hddm.md](references/api-hddm.md) and [api-kabuki-pymc2.md](references/api-kabuki-pymc2.md) |
| Experimental planning | Task or sample-size question | Read [experimental-design.md](references/experimental-design.md) |

Handle hybrid requests in this order: simulate → recovery → fit real data →
diagnostics → PPC/model comparison → interpretation.

## Establish the runtime first

0. **Load project-local `env.json` (required).** All important runtime config
   lives in `<project>/env.json` — **never** in a global directory. On every run,
   first check for this file in the project directory:
   - If it exists and is non-empty, load it: read `image_tag` (which dockerHDDM
     version to pull/run), `mount` (container mount point), `dockerhddm_repo`
     (optional local source clone), and `default_mode`.
   - If it is **missing or empty**, generate it with
     `python scripts/init_project.py <project> --mode <mode> [--image-tag <tag>]`,
     then **remind the user to add one line to their `AGENTS.md`** so the agent
     reads it first each time:
     > 每次开始本项目工作前，先阅读项目根目录的 `env.json`，再动手。
   Never copy personal absolute paths into generated projects or into `env.json`.
1. Treat the submodule commits pinned by `dockerhddm_repo` (optional, from
   `env.json`) as authoritative when present; otherwise rely on the pinned image.
   Separate legacy clones are diagnostic history, not the v1.1 runtime. Read
   [source-map.md](references/source-map.md).
2. Run `python scripts/inspect_environment.py` inside the target container or
   environment. It prints the expected tag (`1.1.0`) and warns on mismatch. Do
   not assume an API from a current upstream HDDM installation.
3. **Version policy.** Use the `image_tag` from `env.json` (default
   `hcp4715/hddm:1.1.0`). The skill also supports `1.0.1`, `0.8`, `latest`, and
   other tags — set `image_tag` accordingly (or pass `--image-tag` to init).
   **However, our workflow, API notes, and debugging guidance are validated
   against `1.1.0`.** If you use another tag and hit behavior that conflicts with
   this skill, warn the user that `1.1.0` is authoritative and report the
   discrepancy. See [docker-runtime.md](references/docker-runtime.md).
   For **launch configurations** (`--cpus` core limit, multiple `-v` mounts,
   custom `-p` host ports, `-w` working dir, interactive vs `bash -lc` batch) and
   the **two usage modes** (Jupyter notebook + how to connect to the kernel vs
   Bash + plain Python scripts), read the "Launch configurations" and "Two usage
   modes" sections of [docker-runtime.md](references/docker-runtime.md).

## Initialize projects

Run (this always writes a project-local `env.json`; pass `--image-tag` to use
a non-default dockerHDDM version):

```bash
python scripts/init_project.py <project-dir> --mode inference
python scripts/init_project.py <project-dir> --mode simulation
python scripts/init_project.py <project-dir> --mode hybrid
python scripts/init_project.py <project-dir> --mode hybrid --image-tag hcp4715/hddm:1.0.1
```

Refuse to overwrite a non-empty destination that already has `env.json`. For an
existing project that only lacks `env.json`, init bootstraps just that file.
Then tailor `config/model.json`, the data dictionary, and only the scripts
relevant to the user's model.

## Enforce the data contract

- Use one row per trial.
- Require `subj_idx`, `rt`, and `response`, plus every condition/covariate used.
- Require RT in seconds; investigate `max(rt) >= 10` before fitting.
- Document whether `response` means accuracy or physical boundary choice.
- For stimulus coding, require exactly two stimulus levels and document
  `stim_col` plus `split_param`.
- Never silently drop trials. Report missingness, exclusions, outliers,
  per-subject counts, choice/error rates, and RT quantiles.
- Preserve raw data in `data/raw/`; write modeling data to `data/processed/`.

## Generate inference code

Always produce:

1. data validation and descriptive checks;
2. a minimally adequate baseline model;
3. a smoke fit with small draws;
4. production sampling with saved `.db`, `.hddm`, and `.nc` artifacts;
5. convergence diagnostics;
6. absolute fit/PPC before parameter interpretation;
7. recovery or sensitivity analysis for the parameters driving the claim.

Do not put `n_jobs` in `model.sample(...)` for the pinned v1.1 source. Sample
first, then call `model.to_infdata(..., n_jobs=1)` for PPC/log-likelihood. See
[api-kabuki-pymc2.md](references/api-kabuki-pymc2.md).

## Generate simulation/prediction code

Distinguish:

- forward simulation from fixed parameters;
- condition/grid simulation over theory-defined ranges;
- hierarchical synthetic-data generation;
- posterior predictive simulation after fitting;
- parameter recovery and model recovery.

Use `hddm.model_config[model]` as the executable parameter/bounds source. Never
invent ranges or mix standard HDDM boundary scaling with HDDMnn/LAN scaling.
Save ground truth, seed, model name, parameter order, and summaries with every
simulation. Read [simulation-prediction-workflow.md](references/simulation-prediction-workflow.md).

## Debug in layers

Stop at the first failing layer:

1. environment/import/build;
2. file mount and permissions;
3. data schema/coding/units;
4. model construction and parameterization;
5. sampling and trace persistence;
6. InferenceData conversion/PPC/log-likelihood;
7. convergence and parameter recovery;
8. absolute fit and theoretical/experimental assumptions.

For memory problems, run `scripts/estimate_memory.py` before expensive work.
Prefer fewer parallel workers, staged InferenceData construction, bounded
`n_ppc`, subsampled LOO where valid, and disk-backed artifacts. A surviving run
is not evidence of a valid model; continue through recovery and PPC.

## Explain meaning, not only syntax

For each generated model, state:

- which observed data features constrain each parameter;
- why a parameter varies by condition/covariate;
- what is fixed and why;
- what the coding makes a positive/negative parameter mean;
- what diagnostics would falsify the interpretation;
- which conclusions are conditional on the selected model.

Use [experimental-design.md](references/experimental-design.md) when poor task
design, sparse errors, nonstationarity, or insufficient trials may be the real
problem.

## Reference routing

- Curated notebook navigation: [notebook-routing.md](references/notebook-routing.md)
- All 40 converted notebooks: [notebooks/index.md](references/notebooks/index.md)
- Article recommendations: [zhihu-reading-guide.md](references/zhihu-reading-guide.md)
- Literature synthesis: [literature-guide.md](references/literature-guide.md)
- Pan et al. paper text: [pan-2025-dockerhddm.md](references/literature/pan-2025-dockerhddm.md)
- Boag et al. paper text: [boag-2025-experimental-planning.md](references/literature/boag-2025-experimental-planning.md)

Search large references with `rg` before loading whole files.

## Privacy and provenance

- Do not expose names, server hosts, usernames, raw data, credentials, or
  unpublished project-specific model labels from example projects.
- Store runtime config (image tag, mount, optional source repo) in the
  project-local `env.json`; never write personal absolute paths there or into
  generated projects.
- Translate reusable structure into generic roles (`data`, `scripts`, `models`,
  `results`, `figs`, `docs`, `logs`).
- Cite source commit hashes or notebook paths when explaining customized APIs.
- Mark statements from articles as guidance and confirm executable behavior
  against the pinned source.

## Maintain this skill

On every material update:

1. bump the semantic version above;
2. append the change to [version-history.md](references/version-history.md);
3. run the scripts and compile the project template;
4. execute the eval prompts in `evals/evals.json`;
5. update the central skills index through `skill-manager`.
