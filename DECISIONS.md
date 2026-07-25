# Decision log

Format: context, options, decision, rationale, cost, status.
An entry without a cost is incomplete.

---

## D-000: Specification before code
Context: a scoped assignment evaluated on structure and reasoning rather than
volume of code.
Options: (a) grow the product out of a notebook; (b) write the module
specification first and build against it.
Decision: (b).
Rationale: the brief grades engineering quality and deployability explicitly;
notebook-first work historically produces a monolith with no module boundaries.
Cost: the first working hours produce no visible output, and the spec has to be
updated whenever an interface changes.
Status: accepted

---

## D-001: Product framing follows the signal check
Context: The Phase 0 analysis was conducted before any product development was
initiated. None of the 18 variables received a signal verdict, and this was true
at two different seeds. None of the models outperformed the prior baseline under
the paired CV rule. The model-level permutation test placed the gradient
boosting fit inside its own null distribution (p = 0.55 and p = 0.41). The
positive control with a U-shaped relationship was detected with delta = 0.054,
indicating that the screen does work but simply found nothing in this dataset.

Options: (a) a product which identifies the predictors of satisfaction as
outlined in the brief; (b) a product which tests for the existence of such
predictors and tells the user when there are none.

Decision: (b).

Rationale: Option (a) is only correct if the drivers are present, and the
results indicate that they are not. A set of top variables derived from this
data would be a set of noise, and the user would not be able to detect this. The
product does not assert that these variables are unimportant. It asserts that no
relationship is detectable using this method on this data. The positive control
provides a scale for this claim: an effect of approximately delta = 0.05 would
have been found, so any remaining effect is smaller, and an effect of that size
is not useful for decision making.

Cost: The product cannot do what such products are usually built to do. A user
who needs a list of drivers will not receive one. The second question from the
assignment is answered, but the answer remains close to the prior distribution
and cannot be used for decisions regarding individual employees. This may be
unexpected from the reader's perspective.

Status: accepted

---

## D-002: One common statistic for effect ranking and verdicts
Context: the 18 features are a mix of nominal, ordinal and numeric. The first
spec ranked native effect sizes (Cramer's V, eta squared, Spearman's rho) in a
single table and applied a negligibility threshold stated in Cramer's V.
Options: (a) keep per-type native effect sizes as the ranking key and verdict
basis; (b) compute one statistic for every feature regardless of type — the
out-of-fold log-loss improvement of a univariate model over the prior baseline
(`delta_logloss`) — with its own permutation null, and demote native effect
sizes to descriptive columns.
Decision: (b).
Rationale: eta squared 0.04 and Cramer's V 0.04 do not denote the same amount
of association, so ranking them in one table was not meaningful, and the
negligibility threshold was undefined for 9 of the 18 features. A single proper
scoring rule on out-of-fold predictions is comparable across all feature types
and matches the product's currency (probabilities). This also resolves
`dataset_verdict` to a single definition: `signal_present` iff at least one
feature earns `signal` from the common statistic; the model-level permutation
test is diagnostic only and never alters the verdict, so the univariate and
model-level conclusions cannot disagree.
Cost: `delta_logloss` is less familiar to reviewers than standard effect sizes
and needs a sentence of explanation in the UI; the permutation null requires
refitting per permutation, which is the most expensive computation in the
product; and the negligibility line `DELTA_NEGLIGIBLE = 0.01` nats/observation
is a judgement call that must be sanity-checked against the synthetic tests in
Phase 1.
Status: accepted

---

## D-003: `keep_default_na=False` — "None" is a category, not missing data
Context: the strings `"None"` in `Mental_Health_Condition` (1,196 rows) and
`Physical_Activity` (1,629 rows) are meaningful values. pandas' default
`na_values` list converts the literal string "None" to `NaN` on read; the
original spec then "repaired" this with a null-to-`"None"` conversion, treating
a parser artefact as a data property. M2's `most_frequent` imputer would
otherwise have rewritten roughly a third of `Physical_Activity` from "no
exercise" to `"Weekly"`.
Options: (a) keep pandas defaults and convert nulls back to `"None"`
downstream; (b) load with `keep_default_na=False` so the strings never become
`NaN`, and assert after parsing that they survived.
Decision: (b).
Rationale: (a) fixes the symptom in one column and leaves every other
"None"-like string exposed; the verified CSV contains no empty cells, so there
is nothing genuinely missing to represent. The assertion turns a silent data
corruption into a loud failure.
Cost: genuinely missing data, if it ever appears in a future CSV, now arrives
as an empty string rather than `NaN` and must be caught by `validate` as an
unexpected category rather than by pandas' null machinery.
Status: accepted

---

## D-004: Onsite employees stay in the sample
Context: 1,637 of 5,000 rows have `Work_Location == "Onsite"`, and the target
is named `Satisfaction_with_Remote_Work`.
Options: (a) filter to remote and hybrid respondents so the target reads
literally; (b) keep all rows, treat the target as a general satisfaction rating
with the respondent's work arrangement, and model `Work_Location` as a feature.
Decision: (b).
Rationale: `Work_Location` is the one feature that could plausibly carry a real
effect on this target, and dropping a third of the sample would remove the only
meaningful comparison available.
Cost: for onsite respondents the target's interpretation is weaker, and any
conclusion about remote work specifically must be read with that caveat. This
goes into the README limitations.
Status: accepted

---

## D-005: Proportional-odds check via likelihood-ratio comparison
Context: the ordinal logistic regression assumes proportional odds, and the
spec requires the assumption to be tested and recorded. statsmodels ships no
Brant test.
Options: (a) implement the Brant test by hand; (b) a likelihood-ratio
comparison of the ordinal fit against the multinomial fit, which nests it in
degrees of freedom.
Decision: (b).
Rationale: the LR comparison uses fits the product already produces, is one
line to compute and easy to defend; a hand-rolled Brant test is new statistical
code that would itself need validation the project has no budget for.
Cost: the LR comparison is an omnibus check — it flags that the assumption
fails somewhere, without naming the offending coefficient the way Brant's
per-variable statistics would.
Status: accepted

---

## D-006: Form-agnostic encoding in the univariate screen
Context: the common statistic (D-002) fits one univariate model per feature.
The first design encoded features by type — numeric scaled, ordinal as integer
codes, categorical one-hot — which spends the fewest degrees of freedom.
Options: (a) keep per-type encoding; (b) encode every feature as one-hot over
its levels: categorical and ordinal natively, numeric after quantile binning
into `N_UNIVARIATE_BINS` (5) bins.
Decision: (b).
Rationale: a single-column encoding only detects monotone log-odds trends, so
a non-monotone association — satisfaction peaking at mid-range hours, say —
would be misreported as noise by the very screen whose job is to catch it.
Ordinal encoding remains in `model.py`, where interpretability matters; this
change affects the screening statistic only.
Cost: more degrees of freedom per feature, so somewhat less power against
genuinely monotone effects, and `delta_logloss` for useless features tends to
land slightly below zero rather than at zero (expected: out-of-fold log loss
penalises unhelpful degrees of freedom). Quantile binning discards within-bin
variation for numeric features, and `N_UNIVARIATE_BINS` is one more judgement
constant to sanity-check against the Phase 1 synthetic tests.
Status: accepted

---

## D-007: Model-level permutation test runs on the gradient boosting model
Context: the verify-signal skill said to permutation-test "the selected
model". On this dataset the selection rule picks the prior dummy, and a
dummy's permutation test is degenerate by construction: the prior is invariant
under target shuffles, so every null draw equals the observed value and p = 1
tells you nothing. The spec implicitly assumed a non-degenerate selected
model.
Options: (a) follow the wording and report a vacuous test; (b) run the
diagnostic on the gradient boosting model, the only fitted model able to
express interactions — the case the diagnostic exists to guard.
Decision: (b), ratified after the Phase 0 run.
Rationale: the test's purpose (SPEC M3 known limitation) is to catch
multivariate structure the univariate screen cannot see; that purpose selects
the model, not the selection rule.
Cost: when the dummy wins selection, the diagnostic describes a model the
product will not ship, and the ModelCard must say which model it describes.
Status: accepted

---

## D-008: Permutation p-values use the add-one convention
Context: the spike computed permutation p as k/N — the share of null draws at
least as extreme as the observed statistic — which can return exactly zero.
Options: (a) k/N; (b) (1 + k) / (1 + N) (Phipson-Smyth), counting the observed
statistic as one draw under its own null.
Decision: (b), applied everywhere a permutation p is computed.
Rationale: the observed statistic is itself a draw under the null; a p-value
of exactly zero asserts an impossibility the test cannot support.
Cost: the attainable floor is 1/(N+1) — with N_PERM = 500 that is 0.002, and
after BH across 18 features the smallest achievable adjusted p is about
0.036. N_PERM cannot be reduced much without making `signal` unreachable at
ALPHA = 0.05.
Status: accepted

---

## D-009: "2 CV standard deviations" is measured on paired per-fold differences
Context: the spec said "beats baseline by more than BASELINE_SD_MULTIPLIER CV
standard deviations" without saying whose standard deviation; the spike used
the candidate model's own.
Options: (a) the candidate model's CV std; (b) the baseline's; (c) the std
(ddof=1) of the paired per-fold differences on the shared folds.
Decision: (c), for both the baseline comparison and the model-selection tie
rule.
Rationale: the folds are shared, so fold-to-fold difficulty is common to both
models; the paired estimate removes that shared variance and measures what the
rule actually asks about — the stability of the gap.
Cost: with N_SPLITS = 5 the paired std is estimated from five numbers and is
itself noisy; the rule remains a heuristic gate, not a formal test.
Status: accepted

---

## D-010: N_PERM raised from 500 to 2000
Context: with N_PERM = 500 and the add-one convention (D-008), the smallest
achievable raw permutation p is 1/501, so the smallest achievable BH-adjusted
p across 18 features is 18/501 = 0.036 against ALPHA = 0.05 — the design could
only ever declare `signal` at the exact resolution floor.
Options: (a) keep 500 and accept that `signal` requires the observed statistic
to beat every single permutation; (b) raise N_PERM to 2000, dropping the
adjusted-p floor to 18/2001 = 0.009.
Decision: (b).
Rationale: this corrects the test's resolution so that a true signal has
headroom below ALPHA after correction. It is recorded as a correction to the
design, not as tuning toward a desired outcome: the change was made after the
verdict was already `no_detectable_signal` and can only make `signal` easier
to reach, i.e. it works against the observed result, not for it.
Cost: the per-feature screening is four times slower — measured verify-signal
runtime is 16-25 minutes on the development machine depending on load, nearly
all of it permutation refits — and the bootstrap and permutation counts are no
longer the same number, which is mildly confusing to read in the config table.
Status: accepted

---

## D-011: stats.py rejects unprepared frames instead of guessing types
Context: `stats.py` infers feature typing from dtypes (numeric / ordered
categorical / unordered categorical) while SCHEMA is the declared single
source of truth. The inference is correct exactly when the frame has been
through `prepare()`, which assigns dtypes from SCHEMA. On a raw frame every
string column is object dtype and an ordinal would silently be treated as
nominal — the native effect would be mislabelled (chi-square instead of
Spearman) with no error.
Options: (a) document the implicit dependency on `prepare()` as a known
constraint; (b) add a guard: an object-dtype feature column raises
`ValueError` telling the caller to run `prepare()`.
Decision: (b).
Rationale: a documented constraint fails silently when forgotten; a guard
fails loudly at the entry point, and the failure message contains the fix.
Cost: `signal_report` can no longer screen ad-hoc frames with raw string
columns — synthetic tests and callers must build properly typed columns,
which is a small amount of ceremony per call site.
Status: accepted

---

## D-012: The "fully empty feature" edge case is out of scope
Context: SPEC M2's edge-case table promised "Fully empty feature -> excluded
from the preprocessor, noted in the report". The Phase 4 review found the row
had neither implementation nor test.
Options: (a) implement data-driven column exclusion inside the preprocessor;
(b) strike the row as unreachable in this product.
Decision: (b).
Rationale: M1's load assertion and the committed CSV guarantee no missing
values at training time, and prediction payloads are single rows whose missing
fields are filled per-field with flags — there is no path on which a fully
empty column reaches the preprocessor. Exclusion logic would be dead code
guarding an impossible state.
Cost: if a future dataset replaces the committed CSV, an all-empty column
would surface as imputer behaviour (constant fill) rather than a loud
exclusion, and this decision must be revisited then.
Status: accepted

---

## D-013: Ship the precomputed artifacts in git and in the image
Context: SPEC M7 lets the API train on first start if the model artefact is
absent, so the reviewer never trains by hand. But the artifacts were gitignored
and .dockerignore excluded the whole artifacts directory, so a fresh clone had
nothing to bake in and the container regenerated everything on first start —
a measured 62-minute cold `docker compose up`. That defeats the one-command
deployability the packaging exists to provide.
Options: (a) keep regenerating on first start and document the hour-long wait;
(b) commit the three product artifacts (model.joblib, model_cards.json,
signal_report.json), admit them through .dockerignore, COPY them into the
image, and let the entrypoint regenerate only when they are genuinely absent.
Decision: (b).
Rationale: the artifacts are seed-deterministic and reproducible from the CLI
(verified in Phase 2), so committing them caches a reproducible build output
rather than hiding a manual step. Cold start drops from an hour to the build
time. The first-start regeneration path is retained unchanged as the fallback
SPEC M7 requires, so deleting the artifacts still works.
Cost: git now carries ~1.2 MB of generated output (the signal report's null
draws dominate), and a committed artifact can drift from the code that
produced it — a reviewer who changes stats.py or model.py and does not
regenerate would see stale numbers. Mitigation: the artifacts are reproducible
with `uv run python -m rwsat.cli verify-signal` / `train`, and the entrypoint
regenerates when the baked files are absent.

Follow-up (diagnosis): after baking the artifacts, a clean `docker compose up`
still measured 62 minutes. Isolating it showed the image builds in ~5 minutes
and the container is healthy in ~4 seconds run directly — the lost hour was the
Phase-4 named volume mounted at /app/artifacts, whose init masked the baked
files under cold Docker Desktop I/O and gated the health-dependent UI. The
volume was there to persist runtime-generated artifacts, which no longer exist
now that they ship in the image, so it was removed entirely. With no volume the
container reads the baked artifacts directly; the earlier root-owned-bind-mount
permission concern is moot because nothing writes to a mount in the normal path.
Status: accepted

---

---

## D-014: Ordinal vs multinomial model
Context: the target variable is ordered as dissatisfied < neutral < satisfied.
Options: (a) a multinomial model, which ignores this order; (b) an ordinal
logistic model, which preserves it.
Decision: (b), as the main approach.
Rationale: the ordinal model uses one coefficient per feature and preserves the
order of the target categories. A multinomial model would require two separate
sets of coefficients and would lose the information contained in the ordering of
the responses.
Cost: the proportional-odds assumption has to be checked (covered by D-005). If
the assumption is violated, the multinomial model is kept as a fallback option.
Status: accepted

---

## D-015: Permutation importance instead of `feature_importances_`
Context: feature importance is needed for the model card.
Options: (a) the built-in `feature_importances_` values from tree-based models;
(b) permutation importance with a confidence interval.
Decision: (b), permutation importance only.
Rationale: tree-based feature importance can be non-zero even for pure noise and
can be biased towards features with more levels. Permutation importance instead
measures how much the model's performance changes when a feature is randomly
shuffled, which gives a more direct measure of the feature's contribution to the
model.
Cost: higher computational time, because permutation importance requires
additional model evaluations. It also still describes the model rather than
proving that the feature has a real relationship with the data. This limitation
was confirmed by Gate 3, where `Years_of_Experience` had high model importance
but was not accepted as a reliable signal.
Status: accepted

---

## D-016: Streamlit instead of a JavaScript frontend
Context: the product requires a user interface.
Options: (a) a full frontend using React or another JavaScript framework;
(b) Streamlit.
Decision: (b).
Rationale: with limited time and a single developer, building a full frontend
would use time that was more important for the methodology and documentation.
Streamlit was sufficient to provide the required interface without adding
unnecessary frontend development work.
Cost: a simpler interface compared with a React-based application, with fewer
options for detailed visual customisation.
Status: accepted

---

## D-017: Separate API and UI
Context: the prediction logic needed a clear place in the system.
Options: (a) put everything into one Streamlit script; (b) separate the
prediction service from the user interface using FastAPI and a thin Streamlit UI.
Decision: (b), keep them separate.
Rationale: the prediction logic is treated as a service that could be called by
another interface or application in the future. The UI therefore only handles
interaction and presentation, while the API is responsible for prediction. This
separation is also verified by an import test that checks that the UI does not
contain prediction logic.
Cost: a slightly more complex setup, with two processes instead of one. The
application is therefore less simple to start, but the separation makes the
system structure clearer.
Status: accepted

---

## D-018: Dataset committed to git
Context: the project needs a simple way for a reviewer to access the data.
Options: (a) require the reviewer to download the dataset from Kaggle; (b)
include the CSV file directly in the repository.
Decision: (b), commit the dataset to git.
Rationale: the main reason is deployability. A reviewer should be able to clone
the repository and run the product without creating a Kaggle account or
completing an additional data download step. Including the CSV directly makes the
project easier to reproduce and review.
Cost: a binary CSV file is stored in the repository. For this project this was
considered an acceptable trade-off because the dataset is required to run and
evaluate the system.
Status: accepted
