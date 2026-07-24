# SPEC.md — RWSAT Technical Specification

## Context and product framing

**The problem.** An HR analyst wants to know what to invest in so that employees
are satisfied with remote work. The usual path is: correlation matrix, then a
RandomForest, then `feature_importances_`, then "here are your top three drivers."

That path breaks on the arithmetic. With 15+ features and no correction for
multiple comparisons, the probability of finding at least one "significant"
factor when none exists is about 54%. Tree-based importances are not zero on pure
noise — they spread across features in proportion to cardinality. So the standard
pipeline is guaranteed to produce drivers whether or not any exist, and the user
spends budget on a programme that changes nothing.

**The product.** RWSAT verifies before it asserts. It measures each factor's
association with satisfaction on one common scale — the out-of-fold log-loss
improvement of a univariate model over the prior baseline (see M3) — bootstraps
a confidence interval, corrects for multiple comparisons, compares the observed
statistic against a null distribution built by permuting the target, and returns
one of three verdicts: `signal`, `inconclusive`, `noise`. Only then does it
predict — and it shows the prior class distribution next to every prediction so
the user can see how much the model actually adds.

**The dataset caveat.** This is a synthetic dataset of 5,000 generated records.
There is a reasonable suspicion that its features are independent of the target,
i.e. that no learnable signal exists. That is a hypothesis to be tested by code in
Phase 0, not an assumption. The product is designed to be useful under either
outcome. Pre-flight evidence on this point is recorded in the Data quality
section below.

**The onsite rows.** Employees with `Work_Location == "Onsite"` (1,637 of 5,000)
stay in the sample. The target is therefore read as a general satisfaction
rating with the respondent's current work arrangement, and `Work_Location` is
modelled as a feature, never used as a filter. It is the one feature that could
plausibly carry a real effect on this target, and dropping a third of the sample
would remove the only meaningful comparison available. The cost: for onsite
respondents the target's interpretation is weaker, and any conclusion about
remote work specifically must be read with that caveat (D-004). This limitation
goes into the README.

**Audience.** Primary: an HR analyst who makes budget decisions and does not read
p-values. Needs a verdict with an honest confidence attached. Secondary: a data
scientist auditing the work, who needs the methodology and the metrics. Hence two
levels of interface: verdict in large type, methodology one click away.

**Stack rationale.**
- `statsmodels` — the only mature ordinal logistic regression in Python. The
  target is ordered; ignoring that discards information.
- FastAPI separate from Streamlit — prediction is a service, not a button. The
  separation shows the product can be integrated, not just demoed.
- `uv` — reproducible lockfile and fast installs, which serves the "runs by
  following your instructions" criterion.
- Streamlit over React — with a fixed budget, the frontend must not consume time
  that belongs to methodology and documentation.

Each module below is specified with six blocks: user stories, data model, API,
screens, business logic, edge cases. Implementation should require no follow-up
questions.

---

## Data quality — pre-flight findings

Recorded from direct inspection of the raw CSV on 2026-07-24, **before any
statistical testing**. These observations set the prior for Phase 0; they do not
replace it.

- Every categorical column is near-uniform across its levels, including the
  target: `Unsatisfied` 1,677 / `Neutral` 1,648 / `Satisfied` 1,675.
- 1,607 rows (32%) have `Years_of_Experience > Age - 16`, i.e. a working life
  that began before age 16 — including `EMP0021`, age 26 with 33 years of
  experience.
- No cell in the CSV is empty. The only missing-looking values are the literal
  strings `"None"` in `Mental_Health_Condition` (1,196 rows) and
  `Physical_Activity` (1,629 rows), and both are meaningful categories, not
  missing data (see M1).

The internally impossible Age/Experience pairs are independent evidence that the
columns were generated independently of one another: no joint constraint between
columns survived generation, so a joint relationship with the target is unlikely
to have been built in either. This was established before the signal check and
is why the product is designed to remain useful when no signal is found.

---

## Configuration (`src/rwsat/config.py`)

Every threshold, sample size and tunable constant lives in
`src/rwsat/config.py`. The spec references these constants **by name**; no
literal thresholds appear in function signatures or module code.

| Name | Value | Used for |
|---|---|---|
| `SEED` | 42 | sole source for every random operation |
| `RAW_PATH` | `data/raw/Impact_of_Remote_Work_on_Mental_Health.csv` | `load_raw` |
| `TEST_SIZE` | 0.2 | train/test split |
| `N_SPLITS` | 5 | stratified CV folds (univariate OOF statistic and model CV) |
| `N_UNIVARIATE_BINS` | 5 | quantile bins for numeric features in the univariate screen (M3) |
| `N_BOOT` | 2000 | bootstrap resamples for confidence intervals |
| `N_PERM` | 500 | per-feature permutation null |
| `N_PERM_MODEL` | 200 | model-level permutation test |
| `ALPHA` | 0.05 | significance level applied to BH-adjusted p-values |
| `NOISE_PERMUTATION_P` | 0.2 | lower bound for the `noise` verdict |
| `DELTA_NEGLIGIBLE` | 0.01 | negligible out-of-fold log-loss improvement, nats per observation |
| `BASELINE_SD_MULTIPLIER` | 2 | "beats baseline" margin, in CV standard deviations |
| `CONFIDENCE_DEVIATION_THRESHOLD` | 0.10 | high/low split for `Prediction.confidence` |
| `MIN_CATEGORY_N` | 5 | categories below this merge into `Other` for testing |

All bootstrap confidence intervals use the **percentile method**; BCa is
unnecessary at this sample size.

---

## M1 — Data layer (`src/rwsat/data.py`)

### User stories
- As an engineer, I want a single load function so paths and parsing are not
  duplicated across the project.
- As a reviewer, I want an explicit schema so the data contract is visible.
- As an analyst, I want a clear error on schema mismatch, not a crash mid-training.
- As a researcher, I want a reproducible split.

### Data model
`ColumnSpec` (dataclass): `name: str`, `dtype: Literal["int","float","cat","ord","bin"]`,
`role: Literal["feature","target","identifier","dropped"]`,
`categories: list[str] | None`, `order: list[str] | list[int] | None` — integer
orders cover the 1–5 rating columns, which the CSV stores as integers.

`SCHEMA: dict[str, ColumnSpec]` — the single source of truth for columns.

`TARGET = "Satisfaction_with_Remote_Work"`, ordered
`["Unsatisfied", "Neutral", "Satisfied"]`.

The schema below was verified against the CSV on 2026-07-24 (5,000 rows,
20 columns, no empty cells). Every category value found in the CSV is
enumerated; ordinal orders are declared here and nowhere else.

| Column | dtype | role | values / range |
|---|---|---|---|
| `Employee_ID` | cat | identifier (dropped) | `EMP0001`…, unique |
| `Age` | int | feature | 22–60 |
| `Gender` | cat | feature | `Female`, `Male`, `Non-binary`, `Prefer not to say` |
| `Job_Role` | cat | feature | `Data Scientist`, `Designer`, `HR`, `Marketing`, `Project Manager`, `Sales`, `Software Engineer` |
| `Industry` | cat | feature | `Consulting`, `Education`, `Finance`, `Healthcare`, `IT`, `Manufacturing`, `Retail` |
| `Years_of_Experience` | int | feature | 1–35 |
| `Work_Location` | cat | feature | `Hybrid`, `Onsite`, `Remote` |
| `Hours_Worked_Per_Week` | int | feature | 20–60 |
| `Number_of_Virtual_Meetings` | int | feature | 0–15 |
| `Work_Life_Balance_Rating` | ord | feature | 1 < 2 < 3 < 4 < 5 |
| `Stress_Level` | ord | feature | `Low` < `Medium` < `High` |
| `Mental_Health_Condition` | cat | feature | `Anxiety`, `Burnout`, `Depression`, `None` — `"None"` is a meaningful category (no condition), **not** missing data |
| `Access_to_Mental_Health_Resources` | bin | feature | `No`, `Yes` |
| `Productivity_Change` | ord | feature | `Decrease` < `No Change` < `Increase` |
| `Social_Isolation_Rating` | ord | feature | 1 < 2 < 3 < 4 < 5 |
| `Satisfaction_with_Remote_Work` | ord | **target** | `Unsatisfied` < `Neutral` < `Satisfied` |
| `Company_Support_for_Remote_Work` | ord | feature | 1 < 2 < 3 < 4 < 5 |
| `Physical_Activity` | ord | feature | `None` < `Weekly` < `Daily` |
| `Sleep_Quality` | ord | feature | `Poor` < `Average` < `Good` |
| `Region` | cat | feature | `Africa`, `Asia`, `Europe`, `North America`, `Oceania`, `South America` |

18 features, one target, one identifier.

### Public interface
```python
load_raw(path: Path = RAW_PATH) -> pd.DataFrame    # raises on unusable input
validate(df: pd.DataFrame) -> ValidationReport     # returns a report, never raises
prepare(df: pd.DataFrame) -> pd.DataFrame          # drop id, cast types, order categories
split(df, test_size=TEST_SIZE, seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]  # stratified
```
`ValidationReport`: `missing_columns`, `unexpected_columns`, `dtype_mismatches`,
`unexpected_categories`, `null_counts`, `is_valid: bool`.

### Business logic
- **`load_raw` passes `keep_default_na=False`** (and sets no `na_values`). The
  strings `"None"` in `Mental_Health_Condition` (1,196 rows) and
  `Physical_Activity` (1,629 rows) are meaningful values, not missing data.
  pandas' default `na_values` would silently convert them to `NaN`, and M2's
  `most_frequent` imputer would then rewrite roughly a third of
  `Physical_Activity` from "no exercise" to `"Weekly"`. After parsing,
  `load_raw` asserts that the literal string `"None"` survived loading in both
  columns (non-zero counts, zero `NaN`). There is **no** null-to-`"None"`
  conversion anywhere in the codebase — the verified CSV contains no empty
  cells, so nothing needs converting. Genuinely missing data would arrive as an
  empty string and is surfaced by `validate` as an unexpected category.
- **Raise/report boundary.** `load_raw` raises on unusable input: missing file
  (`FileNotFoundError` naming the expected path and the download step), empty
  frame or missing header (`ValueError` before any computation). `validate`
  covers schema-level problems — wrong columns, wrong dtypes, unexpected
  categories — by returning a `ValidationReport`, and never raises; callers
  decide what a failed report means for them.
- `Employee_ID` is always dropped: an identifier is not a feature.
- No imputation happens here — that belongs to the Pipeline in M2, otherwise
  statistics leak from the full sample.
- Ordinal columns get `pd.CategoricalDtype(ordered=True)` with the order
  declared in SCHEMA.
- The split is stratified on the target, seeded from config.

### Edge cases
| Situation | Expected behaviour |
|---|---|
| File missing | `load_raw` raises `FileNotFoundError` naming the expected path and the download step |
| Empty dataframe | `load_raw` raises `ValueError` before any computation |
| `"None"` parsed as `NaN` | `load_raw`'s post-parse assertion fails loudly — this means `keep_default_na` was lost in a refactor |
| Required column absent | `validate` returns `is_valid=False`; training exits with a clear message |
| Category not in SCHEMA (including empty strings) | Logged warning, value preserved, listed in the report |
| Duplicate `Employee_ID` | Reported as a warning; rows are not dropped without an explicit decision |

---

## M2 — Feature layer (`src/rwsat/features.py`)

### User stories
- As an engineer, I want one Pipeline object so preprocessing is identical at
  training and inference time.
- As a reviewer, I want confidence that nothing is fitted on the test set.
- As a UI developer, I want feature names and allowed values to build the form.

### Public interface
```python
build_preprocessor(schema: dict[str, ColumnSpec]) -> ColumnTransformer
feature_metadata(schema) -> list[FeatureMeta]
```
`FeatureMeta`: `name`, `kind` (`numeric`/`categorical`/`ordinal`),
`options: list[str] | None`, `min`/`max`/`default` for numerics, `label: str`.

### Business logic
- Numeric: `SimpleImputer(strategy="median")` -> `StandardScaler`.
- Categorical: `SimpleImputer(strategy="most_frequent")` ->
  `OneHotEncoder(handle_unknown="ignore", drop="first")`.
- Ordinal: `OrdinalEncoder` with explicit `categories=` from SCHEMA. Order is
  declared by hand, never inferred from the data.
- The imputers exist for inference-time payloads with missing fields
  (`POST /predict`), not for the training data: the raw CSV contains no missing
  values once M1 loads it with `keep_default_na=False`. This safety depends on
  M1 — if `"None"` were ever parsed as `NaN`, `most_frequent` would silently
  rewrite it, which is exactly the failure M1's assertion guards against.
- Every transformer sits inside the `Pipeline` passed to cross-validation. No
  `.fit()` is called outside a training fold.

### Edge cases
| Situation | Behaviour |
|---|---|
| Unknown category at inference | `handle_unknown="ignore"`; prediction returned with flag `unknown_category` |
| Value outside the training range | Prediction returned with flag `out_of_range`, confidence lowered |
| Fully empty feature | Excluded from the preprocessor, noted in the report |

### Test requirement
A test must prove absence of leakage: fit the Pipeline on train and assert the
scaler's statistics match train-only statistics, not full-sample statistics.

---

## M3 — Inference layer (`src/rwsat/stats.py`)

The core of the product. Answers question 1.

### User stories
- As an HR analyst, I want to know whether a factor affects satisfaction and how
  much I can trust that.
- As an HR analyst, I want factors ranked on one scale that means the same thing
  for every factor.
- As a data scientist, I want to see the multiple-comparison correction.
- As a data scientist, I want the observed statistic placed against a null
  distribution, not a single p-value.
- As any user, I want the words "indistinguishable from noise" when that is true.

### The common statistic

The features are a mix of nominal, ordinal and numeric. Native effect sizes
(Cramer's V, eta squared, Spearman's rho) do not share a scale — eta squared
0.04 and Cramer's V 0.04 do not denote the same amount of association — so
ranking them in one table was not meaningful, and the negligibility threshold
was undefined for 9 of the 18 features (D-002). One statistic is therefore
computed for every feature regardless of type:

**`delta_logloss` — the out-of-fold log-loss improvement of a univariate model
over the prior baseline**, in nats per observation. Positive means the feature
carries information about the target.

Computation for a feature X:
1. Baseline: the prior — training-fold class frequencies predicted for every
   row (equivalent to `DummyClassifier(strategy="prior")`).
2. Univariate model: a `Pipeline` of a **form-agnostic one-hot encoding** and
   multinomial `LogisticRegression` with default regularisation, seeded from
   config. Every feature is encoded as one-hot over its levels: categorical
   and ordinal features natively, numeric features after quantile binning into
   `N_UNIVARIATE_BINS` bins (the binner is fitted inside the CV fold like
   every other transformer). A single-column encoding — ordinal codes or a
   scaled numeric — only detects monotone log-odds trends, so a non-monotone
   association would be misreported as noise (D-006). Ordinal encoding remains
   in M4's model layer, where interpretability matters; this choice affects
   the screening statistic only.
3. Stratified `N_SPLITS`-fold CV on the **training split only** — the held-out
   test set never enters univariate screening. Collect out-of-fold predicted
   probabilities for both models.
4. `delta_logloss` = mean over rows of (per-row baseline log loss − per-row
   model log loss).

`delta_logloss` may be **negative** for a useless feature, and that is
expected: out-of-fold log loss penalises degrees of freedom that do not help,
so a feature carrying no information tends to land slightly below zero rather
than at zero. A negative value is not an error and needs no special handling —
it simply feeds the `noise` verdict.

Uncertainty and significance:
- CI: bootstrap over the per-row out-of-fold loss differences, `N_BOOT`
  resamples, percentile method. No refitting inside the bootstrap.
- Null distribution: permute the target `N_PERM` times and recompute the full
  out-of-fold `delta_logloss` each time (refit per permutation).
  `permutation_p` = share of permutations with a delta >= the observed one.
- Benjamini-Hochberg across all features, applied to the permutation p-values.

This statistic is the ranking key for `GET /effects` and the **sole basis of
the verdict**. The native effect sizes are still computed and reported in
`EffectResult` as descriptive columns, but they never drive sorting or
verdicts.

### Data model
```python
@dataclass
class EffectResult:
    feature: str
    # --- the common statistic: sole basis for ranking and verdict ---
    delta_logloss: float         # OOF log-loss improvement over the prior baseline
    delta_ci_low: float          # bootstrap percentile CI, N_BOOT resamples
    delta_ci_high: float
    permutation_p: float         # share of null deltas >= observed
    permutation_p_adj: float     # Benjamini-Hochberg across all features
    null_mean: float             # of the permutation null of delta_logloss
    null_p95: float
    verdict: Literal["signal", "inconclusive", "noise"]
    explanation: str             # one plain-English sentence, no jargon
    n: int
    # --- native effect size: descriptive only ---
    native_test: Literal["chi2", "kruskal", "spearman"]
    native_effect_size: float    # Cramer's V | eta^2 | rho
    native_effect_name: str

@dataclass
class SignalReport:
    effects: list[EffectResult]  # sorted by delta_logloss, descending
    n_features_tested: int
    n_signal: int
    family_wise_note: str
    dataset_verdict: Literal["signal_present", "no_detectable_signal"]
```

### Public interface
```python
effect_for(df, feature, target, n_boot=N_BOOT, n_perm=N_PERM, seed=SEED) -> EffectResult
signal_report(df, target, features=None) -> SignalReport   # df is the training split
permutation_null(df, feature, target, n_perm=N_PERM) -> np.ndarray  # null deltas
```

### Business logic
- Native test selection stays by variable type: categorical -> chi-square with
  Cramer's V; numeric -> Kruskal-Wallis with eta-squared; ordinal -> Spearman
  against the ordered target. Descriptive columns only.
- Benjamini-Hochberg correction is always applied across the full feature family,
  even when a single feature is requested.
- Verdict rules, all on the common statistic (constants from config):
  - `signal`: `permutation_p_adj < ALPHA` AND `delta_ci_low > DELTA_NEGLIGIBLE`.
  - `noise`: `permutation_p > NOISE_PERMUTATION_P` AND
    `delta_ci_high < DELTA_NEGLIGIBLE`.
  - `inconclusive`: everything else. Never collapse this into a binary.
- **Dataset verdict — single definition and precedence.** `dataset_verdict` is
  `signal_present` iff at least one feature's verdict is `signal` under the
  common statistic; otherwise `no_detectable_signal`. The per-feature common
  statistic is the sole authority. The model-level permutation test (M4) is a
  diagnostic: it appears in the ModelCard and **never alters**
  `dataset_verdict`. If the two appear to conflict — the full model beats its
  permutation null while no single feature earns `signal`, or the reverse —
  the conflict is recorded as a warning in the ModelCard and investigated at
  Gate 3; the dataset verdict does not move. The univariate and model-level
  conclusions therefore cannot disagree about the product's headline verdict,
  because only one of them defines it.
- **Known limitation — interactions.** The univariate screen cannot detect
  interaction effects by construction: each feature is screened alone. The
  model-level permutation test in the ModelCard is the safeguard for that
  case — a multivariate model can beat its permutation null even when no
  single feature earns `signal`. Under the precedence rule such a
  disagreement is a Gate 3 finding to investigate and document, not a
  contradiction in the product. This limitation goes into the README.

### Edge cases
| Situation | Behaviour |
|---|---|
| Constant feature | Univariate model equals the baseline; `delta_logloss = 0`, verdict `noise`, flagged `constant` |
| Category with n < `MIN_CATEGORY_N` | Merged into `Other`, recorded in the result |
| `NaN` in a feature (should not occur after M1) | Row excluded; `n` reflects the actual sample used; warning attached |
| `N_PERM` too small for a precise p | `permutation_p` returned with a note on the `1/N_PERM` resolution |

---

## M4 — Model layer (`src/rwsat/model.py`)

Answers question 2.

### User stories
- As a user, I want a prediction for a specific employee.
- As a user, I want to know whether the model beats simple guessing.
- As a data scientist, I want calibration, not just accuracy.
- As an engineer, I want reproducible training and a serialisable artefact.

### Data model
```python
@dataclass
class ModelCard:
    model_name: str
    trained_at: datetime
    n_train: int
    n_test: int
    metrics: dict[str, float]          # log_loss (primary), balanced_accuracy (secondary),
                                       # accuracy, macro_f1, brier (multiclass)
    baseline_metrics: dict[str, float]
    lift_over_baseline: dict[str, float]
    cv_mean: float                     # of log loss, the primary metric
    cv_std: float
    permutation_test_p: float          # model log loss against N_PERM_MODEL permuted targets
    is_better_than_baseline: bool
    calibration: dict[str, list[tuple[float, float]]]  # one-vs-rest curve per class
    proportional_odds_ok: bool | None
    feature_effects: list[dict]        # permutation importance with CIs
    warnings: list[str]

@dataclass
class Prediction:
    probabilities: dict[str, float]
    predicted_class: str
    prior_probabilities: dict[str, float]
    max_deviation_from_prior: float
    confidence: Literal["high", "low", "not_better_than_guessing"]
    warning: str | None
    flags: list[str]                   # unknown_category, out_of_range, imputed:<field>
```

### Public interface
```python
train_all(df) -> dict[str, ModelCard]   # dummy, ordinal_logit, gradient_boosting
                                        # + multinomial, only when fitted (diagnostic)
select_model(cards) -> str
save(model, path) / load(path)
predict_one(model, payload: dict) -> Prediction
```

### Business logic
- Three primary models, in this order, plus a diagnostic multinomial model
  fitted only when the proportional-odds assumption is violated or statsmodels
  fails to converge:
  1. `DummyClassifier(strategy="prior")` — mandatory baseline.
  2. Ordinal logistic regression (statsmodels `OrderedModel`) — the primary
     interpretable model. The proportional-odds assumption is tested and the
     result recorded.
  3. `HistGradientBoostingClassifier` — an upper bound on achievable quality,
     not a candidate to win by default.
  - Diagnostic: multinomial logistic regression, fitted under the two
    conditions above, never otherwise.
- **Primary metric: log loss** — a proper scoring rule, and the product returns
  probabilities. Model selection and the model permutation test both run on log
  loss. Balanced accuracy is reported as secondary; accuracy, macro-F1 and the
  multiclass Brier score are reported for completeness.
- Proportional-odds test: likelihood-ratio comparison of the ordinal fit
  against the multinomial fit — statsmodels ships no Brant test (D-005).
- Evaluation: stratified `N_SPLITS`-fold CV on train, then a single measurement
  on the held-out test set. The test set is touched once.
- "Better than baseline" means beating the baseline log loss by more than
  `BASELINE_SD_MULTIPLIER` CV standard deviations.
- Model selection: choose the simplest model whose log loss is statistically
  indistinguishable from the best one.
- Calibration is mandatory: a one-vs-rest calibration curve per class plus the
  multiclass Brier score always go into the card.
- `Prediction.confidence`: `"not_better_than_guessing"` whenever the selected
  model does not beat baseline; otherwise `"high"` when
  `max_deviation_from_prior > CONFIDENCE_DEVIATION_THRESHOLD`, `"low"` below it.
- Feature influence: permutation importance with confidence intervals only.
- Model permutation test: `N_PERM_MODEL` target shuffles, log loss as the
  statistic. Diagnostic only — see the precedence rule in M3: this test never
  alters `dataset_verdict`.

### Edge cases
| Situation | Behaviour |
|---|---|
| No model beats baseline | Training does **not** fail. `is_better_than_baseline=False`, an explicit warning enters the card, the product keeps working |
| Proportional-odds assumption violated | Recorded in the card; the diagnostic multinomial model is fitted for comparison |
| statsmodels fails to converge | Logged; model marked unavailable; the diagnostic multinomial model is used instead |
| Model artefact missing at API start | `/predict` returns 503 with the exact command to run |

---

## M5 — API (`src/rwsat/api.py`, FastAPI)

```
GET  /health         -> 200 {"status","model_loaded","data_loaded"}
GET  /schema         -> 200 {"features":[FeatureMeta],"target":{"name","classes"}}
                        The UI builds its form from this. Hardcoding fields in the
                        frontend is a bug.
GET  /effects        -> 200 SignalReport, ranked by delta_logloss descending
GET  /effects/{name} -> 200 EffectResult | 404
POST /predict        body {"features": {...}}
                     -> 200 Prediction | 422 (per-field reasons) | 503 (not trained)
                        Missing fields are allowed: filled with train median/mode
                        and flagged as "imputed:<field>".
GET  /model-card     -> 200 ModelCard for the selected model | 503
GET  /model-card/all -> 200 all cards (three primary; diagnostic multinomial when fitted)
```

### Business logic
- Every request and response body is a pydantic model.
- Model and signal report load once via `lifespan` into `app.state`.
- Heavy computation (permutation tests) never runs inside request handling; it is
  precomputed and cached to disk.
- 503 responses always include the command that fixes them.
- If the model does not beat baseline, every `/predict` response carries a
  non-empty `warning`.

---

## M6 — UI (`app/main.py`, Streamlit)

The UI contains no business logic and imports no internal modules. HTTP only.

**Screen 1 — Overview.** Sample size, target distribution, the dataset verdict in
large type with a one-paragraph plain-English explanation.

**Screen 2 — Factor Explorer** (question 1). Table of all features ranked by
`delta_logloss` — the out-of-fold log-loss improvement over the prior baseline,
the one number computed the same way for every factor — with its CI, the
BH-adjusted permutation p-value, and the colour-coded verdict. Native effect
sizes (Cramer's V / eta squared / rho) appear as descriptive columns, clearly
labelled as not driving the ranking or the verdict. Clicking a feature reveals
a histogram of the permutation null distribution of `delta_logloss` with a
vertical line at the observed value, plus an explanation of the statistic and
the verdict. This chart is the product's key visual — it conveys "signal versus
noise" without words.

**Screen 3 — Predict** (question 2). Form generated from `GET /schema`. Result
shows three probabilities with the prior distribution in grey alongside. If
`confidence == "not_better_than_guessing"`, a warning appears next to the result,
above the fold, stating plainly that this must not drive decisions about people.

**Screen 4 — Model Card.** All primary models next to baseline (plus the
diagnostic multinomial model when it was fitted), per-class one-vs-rest
calibration curves with the multiclass Brier score, model permutation test,
data validation report, selection rationale.

**Screen 5 — Methodology.** Static markdown: which tests, why the BH correction,
how to read a verdict, known dataset limitations. Written by hand.

Every screen handles four states: loading, error, empty, success. An unreachable
API shows a banner with the startup command, never a traceback.
Verdict colours: `signal` green, `inconclusive` amber, `noise` grey — grey rather
than red, because absence of signal is not an error.

---

## M7 — Packaging

```
Dockerfile      multi-stage, uv, non-root user
compose.yaml    api (8000) and ui (8501); ui waits on the api healthcheck
pyproject.toml  dependencies, with uv.lock committed
Makefile        install / test / lint / train / run
data/raw/*.csv  dataset committed, with LICENSE.md naming the source
```

`docker compose up` must produce a working product on a clean machine with no
further steps. If the model artefact is absent, the API trains it on first start;
the reviewer never runs training manually. No environment variable is required for
the default path. README also documents a Docker-free path via `uv sync`.

### CLI (`src/rwsat/cli.py`)

Three commands, matching the Commands section of CLAUDE.md:

- `uv run python -m rwsat.cli validate` — load the raw CSV via `load_raw` and
  print the `ValidationReport`; exit code 1 when `is_valid` is `False`.
- `uv run python -m rwsat.cli verify-signal` — run the signal-verification
  protocol on the training split, write `artifacts/signal_report.json`, print
  the dataset verdict with its one-paragraph explanation.
- `uv run python -m rwsat.cli train` — train the primary models, run selection,
  serialise the artefact and ModelCards. This is the exact command quoted by
  `/predict`'s 503 response.

### UI API base URL

The UI reads the API base URL from the `RWSAT_API_URL` environment variable.
The in-code default is `http://localhost:8000` (the Docker-free path);
`compose.yaml` sets `RWSAT_API_URL=http://api:8000` for the ui service. Neither
path requires the user to set anything, which keeps CLAUDE.md Rule 6 intact.

---

## M8 — Agent layer (optional, only after M1-M7 are complete)

A conversational interface over the analysis. The agent does not reason about the
data on its own — it calls tools (`get_effect`, `list_effects`, `describe_segment`,
`predict`) and answers strictly from their output. When a tool returns the `noise`
verdict, the agent says the association was not found and does not offer an
alternative explanation. The API key is read from the environment; without it the
tab is hidden and the rest of the product works fully. The agent must never
compromise the deployability criterion.
