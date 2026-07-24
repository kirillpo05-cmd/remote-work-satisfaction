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
association with satisfaction, bootstraps a confidence interval, corrects for
multiple comparisons, compares the observed effect against a null distribution
built by permuting the target, and returns one of three verdicts: `signal`,
`inconclusive`, `noise`. Only then does it predict — and it shows the prior class
distribution next to every prediction so the user can see how much the model
actually adds.

**The dataset caveat.** This is a synthetic dataset of 5,000 generated records.
There is a reasonable suspicion that its features are independent of the target,
i.e. that no learnable signal exists. That is a hypothesis to be tested by code in
Phase 0, not an assumption. The product is designed to be useful under either
outcome.

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
`categories: list[str] | None`, `order: list[str] | None`, `missing_means: str | None`.

`SCHEMA: dict[str, ColumnSpec]` — the single source of truth for columns.

`TARGET = "Satisfaction_with_Remote_Work"`, ordered
`["Unsatisfied", "Neutral", "Satisfied"]`.

Expected columns (verify against the CSV in Phase 0 and correct this list if it
differs): `Employee_ID` (drop), `Age`, `Gender`, `Job_Role`, `Industry`,
`Years_of_Experience`, `Work_Location`, `Hours_Worked_Per_Week`,
`Number_of_Virtual_Meetings`, `Work_Life_Balance_Rating` (ord 1-5),
`Stress_Level` (ord), `Mental_Health_Condition` (nulls mean "no condition"),
`Access_to_Mental_Health_Resources` (bin), `Productivity_Change`,
`Social_Isolation_Rating` (ord 1-5), `Company_Support_for_Remote_Work` (ord 1-5),
`Physical_Activity`, `Sleep_Quality` (ord), `Region`, and the target.

### Public interface
```python
load_raw(path: Path = RAW_PATH) -> pd.DataFrame
validate(df: pd.DataFrame) -> ValidationReport     # returns a report, does not raise
prepare(df: pd.DataFrame) -> pd.DataFrame          # drop id, cast types, order categories
split(df, test_size=0.2, seed=SEED) -> tuple[pd.DataFrame, pd.DataFrame]  # stratified
```
`ValidationReport`: `missing_columns`, `unexpected_columns`, `dtype_mismatches`,
`unexpected_categories`, `null_counts`, `is_valid: bool`.

### Business logic
- `Employee_ID` is always dropped: an identifier is not a feature.
- Nulls in `Mental_Health_Condition` become an explicit `"None"` category. This is
  absence of a condition, not absence of data; imputing the mode would be wrong.
- No other imputation happens here — that belongs to the Pipeline in M2, otherwise
  statistics leak from the full sample.
- Ordinal columns get `pd.CategoricalDtype(ordered=True)` with hand-written order.
- The split is stratified on the target, seeded from config.

### Edge cases
| Situation | Expected behaviour |
|---|---|
| File missing | `FileNotFoundError` naming the expected path and the download step |
| Required column absent | `validate` returns `is_valid=False`; training exits with a clear message |
| Category not in SCHEMA | Logged warning, value preserved, listed in the report |
| Empty dataframe | `ValueError` before any computation |
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
- As an HR analyst, I want factors ranked by strength of association.
- As a data scientist, I want to see the multiple-comparison correction.
- As a data scientist, I want the observed effect placed against a null
  distribution, not a single p-value.
- As any user, I want the words "indistinguishable from noise" when that is true.

### Data model
```python
@dataclass
class EffectResult:
    feature: str
    test: Literal["chi2", "kruskal", "spearman"]
    effect_size: float          # Cramer's V | eta^2 | rho
    effect_name: str
    ci_low: float               # bootstrap, 2000 resamples
    ci_high: float
    p_value_raw: float
    p_value_adj: float          # Benjamini-Hochberg across all features
    permutation_p: float        # share of permutations with effect >= observed
    null_mean: float
    null_p95: float
    verdict: Literal["signal", "inconclusive", "noise"]
    explanation: str            # one plain-English sentence, no jargon
    n: int

@dataclass
class SignalReport:
    effects: list[EffectResult]
    n_features_tested: int
    n_signal: int
    family_wise_note: str
    dataset_verdict: Literal["signal_present", "no_detectable_signal"]
```

### Public interface
```python
effect_for(df, feature, target, n_boot=2000, n_perm=500, seed=SEED) -> EffectResult
signal_report(df, target, features=None) -> SignalReport
permutation_null(df, feature, target, n_perm) -> np.ndarray
```

### Business logic
- Test selection by variable type: categorical -> chi-square with Cramer's V;
  numeric -> Kruskal-Wallis with eta-squared; ordinal -> Spearman against the
  ordered target.
- Benjamini-Hochberg correction is always applied across the full feature family,
  even when a single feature is requested.
- Verdict rules:
  - `signal`: `p_value_adj < 0.05` AND `permutation_p < 0.05` AND the confidence
    interval excludes the negligible-effect threshold (Cramer's V < 0.1).
  - `noise`: `permutation_p > 0.2` AND the upper CI bound is below the
    negligibility threshold.
  - `inconclusive`: everything else. Never collapse this into a binary.
- Dataset verdict is `signal_present` if any feature earns `signal`, else
  `no_detectable_signal`.

### Edge cases
| Situation | Behaviour |
|---|---|
| Constant feature | No test run; verdict `noise`, flagged `constant` |
| Category with n < 5 | Merged into `Other`, recorded in the result |
| Nulls in the feature | Pairwise deletion; `n` reflects the actual sample used |
| n_perm too small for a precise p | `permutation_p` returned with a note on `1/n_perm` resolution |

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
    metrics: dict[str, float]          # accuracy, balanced_accuracy, macro_f1, log_loss, brier
    baseline_metrics: dict[str, float]
    lift_over_baseline: dict[str, float]
    cv_mean: float
    cv_std: float
    permutation_test_p: float          # model significance against permuted targets
    is_better_than_baseline: bool
    calibration: list[tuple[float, float]]
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
select_model(cards) -> str
save(model, path) / load(path)
predict_one(model, payload: dict) -> Prediction
```

### Business logic
- Exactly three models, in this order:
  1. `DummyClassifier(strategy="prior")` — mandatory baseline.
  2. Ordinal logistic regression (statsmodels `OrderedModel`) — the primary
     interpretable model. The proportional-odds assumption is tested and the
     result recorded.
  3. `HistGradientBoostingClassifier` — an upper bound on achievable quality,
     not a candidate to win by default.
- Evaluation: stratified 5-fold CV on train, then a single measurement on the
  held-out test set. The test set is touched once.
- "Better than baseline" means exceeding it by more than 2 CV standard deviations.
- Model selection: choose the simplest model that is statistically
  indistinguishable from the best one.
- Calibration is mandatory. Brier score and calibration curve points always go
  into the card.
- Feature influence: permutation importance with confidence intervals only.

### Edge cases
| Situation | Behaviour |
|---|---|
| No model beats baseline | Training does **not** fail. `is_better_than_baseline=False`, an explicit warning enters the card, the product keeps working |
| Proportional-odds assumption violated | Recorded in the card; a multinomial model is additionally fitted for comparison |
| statsmodels fails to converge | Logged; model marked unavailable; multinomial used instead |
| Model artefact missing at API start | `/predict` returns 503 with the exact command to run |

---

## M5 — API (`src/rwsat/api.py`, FastAPI)

```
GET  /health         -> 200 {"status","model_loaded","data_loaded"}
GET  /schema         -> 200 {"features":[FeatureMeta],"target":{"name","classes"}}
                        The UI builds its form from this. Hardcoding fields in the
                        frontend is a bug.
GET  /effects        -> 200 SignalReport, sorted by effect size
GET  /effects/{name} -> 200 EffectResult | 404
POST /predict        body {"features": {...}}
                     -> 200 Prediction | 422 (per-field reasons) | 503 (not trained)
                        Missing fields are allowed: filled with train median/mode
                        and flagged as "imputed:<field>".
GET  /model-card     -> 200 ModelCard for the selected model | 503
GET  /model-card/all -> 200 all three cards
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

**Screen 2 — Factor Explorer** (question 1). Table of all features: effect size
with CI, adjusted p-value, colour-coded verdict. Clicking a feature reveals a
histogram of the permutation null distribution with a vertical line at the
observed effect, plus an explanation of the test and the verdict. This chart is
the product's key visual — it conveys "signal versus noise" without words.

**Screen 3 — Predict** (question 2). Form generated from `GET /schema`. Result
shows three probabilities with the prior distribution in grey alongside. If
`confidence == "not_better_than_guessing"`, a warning appears next to the result,
above the fold, stating plainly that this must not drive decisions about people.

**Screen 4 — Model Card.** All three models next to baseline, calibration curve,
model permutation test, data validation report, selection rationale.

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

---

## M8 — Agent layer (optional, only after M1-M7 are complete)

A conversational interface over the analysis. The agent does not reason about the
data on its own — it calls tools (`get_effect`, `list_effects`, `describe_segment`,
`predict`) and answers strictly from their output. When a tool returns the `noise`
verdict, the agent says the association was not found and does not offer an
alternative explanation. The API key is read from the environment; without it the
tab is hidden and the rest of the product works fully. The agent must never
compromise the deployability criterion.
