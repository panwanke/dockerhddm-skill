# Experimental design for EAM/HDDM

Based on the converted Boag et al. (2025) expert guide. Confirm exact statements
against the paper text before quoting.

## Task suitability

- Decisions should have a clear stimulus onset and discrete response.
- Evidence and decision settings should be approximately stationary within a
  trial and across trials of the same condition.
- Avoid floor/chance performance and ceiling accuracy.
- Use sufficient practice to stabilize behavior.
- Prefer variable fixation timing and adequate intertrial intervals.
- Avoid trial-wise feedback unless learning is explicitly modeled; block-level
  summaries can support engagement with less nonstationarity.

## Trial and participant planning

- Around 200 trials per condition is a broad starting point for reasonably
  precise individual measurement.
- Gains diminish beyond roughly 500 trials per condition in some simulations.
- Aim for at least about 5% errors; with 200 trials that yields only 10 errors,
  a bare minimum rather than a strong target.
- Simpler models with one varying parameter may work with 50–100 trials per
  condition; variability parameters or rare responses can need thousands.
- Individual-difference studies often need ~80+ participants and ~200 trials
  each; do not treat this as a universal power rule.
- Use parameter-recovery simulations across effect sizes, participant counts,
  trial counts, and error rates for the actual planned model.

## Data recording

Record participant, condition, stimulus, response, RT, session, trial number,
and event timing. Raw data should permit reconstruction of trial composition.

## Screening

Inspect fast guesses, slow contaminants, nonresponses, truncation, RT
distribution shape, time-on-task trends, and noncompliant participants.
Typical fast-cutoff conventions around 150–300 ms are context-dependent; use
psychological plausibility and sensitivity analyses, not automatic deletion.

## Evaluation

- fit individual/trial data rather than group-aggregated means;
- assess relative and absolute fit;
- check choices and full RT range (at least .1/.5/.9 quantiles);
- inspect individual PPC when possible;
- run parameter recovery over a range of true values;
- interpret parameters only after reliable recovery and adequate fit.

## Reporting

Report task timing, trial/participant counts, exclusions, RT and choice
descriptives, units, parameter coding, varied/fixed parameters, priors, fitting
method, convergence, absolute fit, comparison, recovery, and inferential tests.
