---
description: Statistical and modelling correctness rules
paths:
  - "src/rwsat/stats.py"
  - "src/rwsat/model.py"
  - "src/rwsat/features.py"
  - "tests/**/*.py"
---

# ML and statistics hygiene

These rules are not negotiable and are not relaxed to improve a result.

## Leakage
- `split()` is called before any `fit`. No exceptions.
- All transformers live inside the `sklearn.Pipeline` passed to cross-validation.
- The held-out test set is used exactly once, for the final measurement.
- Any feature selection happens inside the fold, never beforehand.

## Claims about influence
- No number becomes a claim without a confidence interval.
- Multiple comparisons are corrected with Benjamini-Hochberg across the full family.
- Tree `feature_importances_` are never used as evidence of influence.
- Ranking and verdicts run on the common statistic (`delta_logloss`, SPEC M3);
  native effect sizes are descriptive only.
- A small permutation p with `delta_logloss` below `DELTA_NEGLIGIBLE` is
  reported as a negligible effect.

## Baseline
- `DummyClassifier(strategy="prior")` is trained first in every experiment.
- Every reported metric appears next to the baseline metric.
- "Beats baseline" means the mean paired per-fold difference exceeds
  `BASELINE_SD_MULTIPLIER` standard deviations of those paired differences
  (shared folds; never either model's own CV std).
- Permutation p-values use the add-one convention (1 + k) / (1 + N); a
  permutation p of exactly zero is a bug, not a result.

## Reproducibility
- The only source of seeds is `src/rwsat/config.py`.
- No random function is called without an explicit `random_state`.
- Rerunning `train` must produce identical metrics.

## Negative results
- Absence of signal is a valid and anticipated outcome. It is not a reason to
  change methodology, add features or tune hyperparameters.
- Hedging phrases such as "the model shows a tendency" are forbidden when
  `p_adj > 0.05`.
