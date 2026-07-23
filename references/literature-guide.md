# Literature guide

## Pan et al. (2025), dockerHDDM

Use for:

- why containerization improves installation and reproducibility;
- Apple Silicon/cross-platform motivation;
- the principled Bayesian workflow enabled by ArviZ;
- Docker pull/run/mount concepts;
- diagnostics, model comparison, PPC, and reporting context.

The paper describes three core improvements: easier reproducible installation,
Apple-chip compatibility, and ArviZ integration. Its workflow predates some
v1.1 API changes; use the current notebooks for executable code.

Full converted text: [pan-2025-dockerhddm.md](literature/pan-2025-dockerhddm.md).

## Boag et al. (2025), experimental task planning

Use for:

- deciding whether a task satisfies EAM assumptions;
- trial timing, feedback, training, and stationarity;
- trial/participant planning;
- error-rate and identifiability concerns;
- exclusions and data-quality checks;
- relative fit, absolute fit, and parameter recovery;
- reporting standards.

High-value quantitative guidance includes ~200 trials per condition as a broad
starting point, diminishing returns around ~500 in some settings, and at least
~5% errors so the least frequent response distribution is constrained. The
paper emphasizes that parameter recovery for the actual design/model overrides
generic rules.

Full converted text:
[boag-2025-experimental-planning.md](literature/boag-2025-experimental-planning.md).

## Joint use

Pan et al. explains the reproducible computational environment and analysis
workflow; Boag et al. explains whether the data-generating experiment makes the
model identifiable and interpretable. Use both before concluding that a
successful chain implies a successful study.
