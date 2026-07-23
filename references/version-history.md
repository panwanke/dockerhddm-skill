# Version history

## 0.2.2 — 2026-07-23

- Covered single-subject, multi-condition truth serialization.
- Removed a recent-pandas-only `groupby.apply` option from recovery metrics for
  broader compatibility with the container stack.

## 0.2.1 — 2026-07-23

- Removed the remaining user-specific environment path from the skill body.
- Fixed synthetic ground-truth serialization for the actual dictionary/list
  return types of `gen_rand_data`.
- Added reusable LOO comparison and recovery-metric scripts after forward
  validation of the generated project.

## 0.2.0 — 2026-07-23

- Corrected the `n_jobs` boundary after source-level validation.
- Distinguished pinned dockerHDDM submodules from older custom clones.
- Added executable project initialization, environment inspection, memory
  estimation, data validation, inference, simulation, recovery, and plotting
  templates.
- Added 40 converted notebook references, two converted papers, experimental
  design guidance, debugging ladder, and Zhihu problem-to-article routing.
- Added privacy and source-precedence rules.

## 0.1.0 — 2026-07-23

- Initialized the skill with `scripts/`, `references/`, `assets/`, and UI
  metadata.
- Registered the planned slash command `/dockerhddm-workflow`.
