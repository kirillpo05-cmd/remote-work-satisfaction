# Working with an AI assistant

The brief asked where judgement was exercised by the author and where the tool
was simply generating. This document answers that. It is written as the work
happens, not reconstructed afterwards.

## How the tooling was used

<!-- Fill in factually: which tools, on which tasks, roughly what share of the
     code was generated, how the context was set up. -->

## How the tooling was configured

This repository ships a `CLAUDE.md`, three specialised subagents, two
glob-scoped rule sets and one project-specific skill. That is a deliberate
investment: an assistant without project context produces locally correct but
architecturally disconnected code. The configuration is part of the submitted
work.

Two configuration decisions worth naming:

- The `reviewer` subagent is configured without `Write` or `Edit`, but it keeps
  `Bash` so it can run tests — and Bash can write files. So this is a default
  constraint, not an enforced guarantee: it lowers the chance the reviewer
  starts fixing instead of reporting, where a fixed problem would disappear
  from the report along with the chance to discuss it. Whether the reviewer
  actually stayed read-only is checked by reading the diff after each review
  pass, not assumed from the toolset.
- Only the statistical and modelling agent runs on the larger model. The service
  and UI layers are mechanical enough that the smaller model handles them, and
  the cost difference is real over a long session.

---

## Episodes

<!-- Format:

## Episode N: <title>
Task:
What the assistant proposed:
Why it did not fit:
What was done instead:
What it says:

Aim for at least five. Likely candidates, based on how these pipelines usually
get generated:

1. Feature influence: RandomForest + feature_importances_ instead of permutation
   importance with confidence intervals.
2. Splitting: preprocessing applied before the train/test split.
3. Metrics: accuracy reported with no baseline and no calibration.
4. Reaction to a weak result: a suggestion to tune hyperparameters rather than
   conclude that no signal exists.
5. Stale APIs: statsmodels OrderedModel usage, or FastAPI on_event instead of
   lifespan.
6. Missing values: the original spec assumed nulls in Mental_Health_Condition
   and converted them to a "None" category; pre-flight inspection showed the
   CSV has no empty cells and pandas' default na_values was manufacturing the
   nulls from literal "None" strings (fixed with keep_default_na=False, D-003).
7. README prose: evaluative adjectives instead of numbers.
-->

## Episode 1: permutation p-values that could reach zero
Task: Phase 0 spike, permutation tests for the common statistic and the model.
What the assistant proposed: p = k/N — the share of null draws at least as
extreme as the observed statistic — matching the spec's literal wording at the
time.
Why it did not fit: the observed statistic is itself a draw under the null; a
p-value of exactly zero asserts an impossibility the test cannot support, and
the spike duly printed perm_p=0.000 for Work_Location in the smoke run.
What was done instead: (1 + k) / (1 + N) everywhere (D-008), with the spec,
skill and rules updated to match.
What it says: the assistant implemented the spec as written rather than the
statistically correct convention; the human review caught what the spec
missed. Wording in a spec gets executed literally.

## Episode 2: whose standard deviation is "2 CV standard deviations"?
Task: the "beats baseline" and model-selection rules in the Phase 0 spike.
What the assistant proposed: compare the mean log-loss gap against two CV
standard deviations of the candidate model's own fold scores, and it flagged
the ambiguity in its report rather than resolving it.
Why it did not fit: the folds are shared between models, so fold-to-fold
difficulty is common variance; a model's own std conflates that shared
variance with the stability of the gap being tested.
What was done instead: the std (ddof=1) of the paired per-fold differences
(D-009), for both rules.
What it says: when the assistant hits an underspecified rule it picks a
defensible reading and flags it — the flag worked, but the default it picked
was not the best one. Ambiguities in specs are cheaper to resolve before the
first implementation than after.

## Episode 3: the degenerate permutation test the spec didn't foresee
Task: the model-level permutation test in the Phase 0 spike.
What the assistant proposed: deviating from the skill's instruction to test
"the selected model", because the selection rule picks the prior dummy on this
dataset and a dummy's permutation test is degenerate (the prior is invariant
under target shuffles; p = 1 by construction). It ran the test on the gradient
boosting model instead and asked for ratification.
Why it did not fit (the original wording): the skill assumed a non-degenerate
selected model, which is exactly the assumption a no-signal dataset violates.
What was done instead: the deviation was ratified and recorded (D-007); spec
and skill updated.
What it says: the useful failure mode — the assistant followed the intent of
the diagnostic rather than its letter, and surfaced the conflict instead of
silently picking either.

---

## Where the tooling helped most

<!-- Be specific and honest. Usually: boilerplate, edge-case tests, Docker,
     format conversions. -->

## Where it was useless or actively wrong

<!-- Be specific and honest. What had to be rewritten from scratch, and why. -->
