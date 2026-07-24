---
name: verify-signal
description: Protocol for verifying whether predictive signal exists in the dataset. Run first, before any modelling or UI work, and again after any preprocessing change.
---

# Skill: verify signal

This protocol decides the product framing for the whole project. It runs before
the model layer and before the UI.

## Steps

**1. Baseline.** Train `DummyClassifier(strategy="prior")` and
`DummyClassifier(strategy="stratified")`. Record accuracy, balanced accuracy,
macro-F1 and log loss. This is the reference point for everything that follows.

**2. Univariate associations.** For every feature, compute the effect size with a
bootstrap CI and a p-value. Apply the Benjamini-Hochberg correction across the
full family. Save the table.

**3. Models.** Train the ordinal logistic regression and
`HistGradientBoostingClassifier` under stratified 5-fold CV. Compare against
baseline on every metric.

**4. Model permutation test.** Shuffle the target at least 200 times, retrain the
selected model, collect the null distribution of the metric, and compute the share
of shuffles that did at least as well as the real fit.

**5. Verdict.** Observed metric inside the null distribution ->
`no_detectable_signal`. Substantially to the right of it -> `signal_present`.

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
