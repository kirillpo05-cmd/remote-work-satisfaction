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
Context: <fill in with the actual Phase 0 verdict>
Options: (a) a driver-explanation product; (b) a verification product that
protects the user from a false conclusion.
Decision: <fill in>
Rationale: <fill in>
Cost: <fill in>
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

<!-- Decisions that must be recorded as the build proceeds:

- Ordinal versus multinomial target treatment
- Permutation importance instead of feature_importances_
- Streamlit instead of a JS frontend
- Splitting the API from the UI
- Committing the dataset instead of requiring Kaggle credentials
- The "simplest model that ties" selection rule
-->
