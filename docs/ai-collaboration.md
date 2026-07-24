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

---

## Where the tooling helped most

<!-- Be specific and honest. Usually: boilerplate, edge-case tests, Docker,
     format conversions. -->

## Where it was useless or actively wrong

<!-- Be specific and honest. What had to be rewritten from scratch, and why. -->
