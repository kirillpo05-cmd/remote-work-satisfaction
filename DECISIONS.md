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
Cost: the per-feature screening is four times slower (about 45 minutes of the
full protocol's runtime is permutation refits), and the bootstrap and
permutation counts are no longer the same number, which is mildly confusing to
read in the config table.
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

<!-- Decisions that must be recorded as the build proceeds:

- Ordinal versus multinomial target treatment
- Permutation importance instead of feature_importances_
- Streamlit instead of a JS frontend
- Splitting the API from the UI
- Committing the dataset instead of requiring Kaggle credentials
- The "simplest model that ties" selection rule
-->
