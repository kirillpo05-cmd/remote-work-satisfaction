---
name: verify-signal
description: Protocol for verifying whether predictive signal exists in the dataset. Run first, before any modelling or UI work, and again after any preprocessing change.
---

# Skill: verify signal

This protocol decides the product framing for the whole project. It runs before
the model layer and before the UI. In Phase 0 it is executed by the explicit
spike `scripts/spike_signal_check.py` (exempt from CLAUDE.md Rule 1; deleted or
absorbed into spec'd modules before Phase 2 ends), later by
`uv run python -m rwsat.cli verify-signal`.

## Steps

**1. Baseline.** Train `DummyClassifier(strategy="prior")` and
`DummyClassifier(strategy="stratified")`. Record log loss (primary), balanced
accuracy (secondary), accuracy and macro-F1. This is the reference point for
everything that follows.

**2. Common statistic per feature.** For every feature, compute `delta_logloss`
— the out-of-fold log-loss improvement of a univariate model over the prior
baseline (SPEC M3) — with a bootstrap percentile CI and a permutation p-value.
Apply the Benjamini-Hochberg correction across the full family. Compute native
effect sizes as descriptive columns. Save the table.

**3. Models.** Train the ordinal logistic regression and
`HistGradientBoostingClassifier` under stratified `N_SPLITS`-fold CV. Compare
against baseline on every metric; log loss decides.

**4. Model permutation test.** Shuffle the target `N_PERM_MODEL` times, retrain
the selected model, collect the null distribution of log loss, and compute the
share of shuffles that did at least as well as the real fit.

**5. Verdict.** `dataset_verdict` follows SPEC M3, and only SPEC M3:
`signal_present` iff at least one feature earns `signal` from the common
statistic, else `no_detectable_signal`. The model permutation test from step 4
is diagnostic corroboration; per the precedence rule it never overrides the
per-feature result — a conflict between the two is recorded as a finding, not
resolved by picking the friendlier one.

## Forbidden on a negative result

- Tuning hyperparameters until the metric moves
- Switching to a more flattering metric
- Adding feature interactions in the hope of finding an association
- Collapsing to a binary target to inflate accuracy
- Softening the wording of the conclusion

A negative result answers the assignment's question. It is not an obstacle.

## Output

- `artifacts/signal_report.json` — machine-readable report
- A `DECISIONS.md` entry recording the verdict and its consequences for scope
- A draft "Headline finding" section for the README
