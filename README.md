# RWSAT — Remote Work Satisfaction Explorer

A data product built on the Kaggle dataset
[`waqi786/remote-work-and-mental-health`](https://www.kaggle.com/datasets/waqi786/remote-work-and-mental-health)
(5,000 rows, 20 columns). It answers two questions:

1. **Which factors drive employee satisfaction with remote work?**
2. **Given an employee's attributes, what satisfaction level do we predict?**

Its guiding principle is honesty over accuracy: it separates signal from noise
and says so plainly when a conclusion is not statistically supported. Every
effect and every prediction ships with a measure of uncertainty. Predictive
accuracy is explicitly not a success criterion; defensible reasoning is.

---

## Headline finding

None of the 18 factors showed a relationship with satisfaction that remained after a proper validation. The correct conclusion is not that there is no relationship, but that no relationship was detected in these data at the sensitivity of the method used. The strongest candidate was `Work_Location`, but after correction its p-value was 0.198, and its effect was 13 times below the threshold at which it would be considered meaningful. Neither model performed better than the baseline: the dummy baseline had a score of 1.0986, while the ordinal model scored 1.1046 and the boosting model scored 1.2289. Both models were therefore worse than predicting according to the population frequencies.

This result does not indicate that the detector is broken. When tested on artificial data containing a predefined U-shaped relationship, the same method correctly returned "signal". When tested on pure noise, it returned "noise". This means that the absence of a detected signal is a property of the data rather than a failure of the detection method. The method was able to detect relationships with an effect size of around 0.05. Since no such relationship was found, any remaining relationship is weaker than this level and is too small to support a practical decision.

The structure of the dataset also provides a possible explanation for this result. The dataset is synthetic, and the columns appear to have been generated independently of each other. In addition, 32% of the rows contain physically impossible combinations of values. For example, one employee is 26 years old but has 33 years of work experience. This can be identified before applying any statistical method and suggests that the variables are not consistent with each other.

This result answers both questions of the task. For the first question, no influencing factors were found, and the analysis provides evidence for this conclusion rather than simply failing to find a signal. For the second question, prediction is still possible, but its performance is equivalent to using the population distribution. The product therefore reports this limitation instead of presenting noise as a confident prediction. This prevents the system from producing a result that may look convincing but is not supported by the data.

---

## Quick start

### Docker (one command)

```bash
docker compose up
```

Then open the UI at **http://localhost:8501** (the API is on
**http://localhost:8000**). No API keys, no environment variables.

**Honest timing:** the **first `docker compose up` takes about 5 minutes**, and
that time is almost entirely the one-time image build (installing the scientific
Python stack). Measured on a settled machine from a fresh clone: **336 seconds
to a working UI reachable from the host**, of which 326s was the build and ~10s
was startup. Every later `docker compose up` starts in **seconds**. The
precomputed analysis artifacts ship in the image, so the API never retrains on
start — it serves them directly (D-013).

### Docker-free (uv)

The committed artifacts mean the app runs without any training step:

```bash
uv sync                                        # install the locked environment
uv run uvicorn rwsat.api:app --port 8000       # terminal 1: the API
uv run streamlit run app/main.py --server.port 8501   # terminal 2: the UI
```

To regenerate the analysis yourself (reproducible from a fixed seed):

```bash
uv run python -m rwsat.cli validate         # check the raw CSV against the schema
uv run python -m rwsat.cli verify-signal    # rebuild artifacts/signal_report.json
uv run python -m rwsat.cli train            # rebuild the model + model cards
```

`verify-signal` runs 2,000 permutations per feature and takes roughly 15–25
minutes; `train` adds the model-level permutation test (a further ~10–15
minutes). Neither is needed to run the product — the artifacts are committed.

---

## Architecture

```
data/raw/*.csv  ->  src/rwsat/data.py      schema, validation, stratified split
                ->  src/rwsat/features.py  sklearn Pipeline, no leakage
                ->  src/rwsat/stats.py     the common statistic, permutation nulls, verdicts
                ->  src/rwsat/model.py     training, calibration, model cards
                ->  src/rwsat/api.py       FastAPI service
                ->  src/rwsat/cli.py       validate / verify-signal / train
                ->  app/main.py            Streamlit UI, talks to the API only
```

One module, one layer. The dependency arrow points one way: the UI knows
nothing about the statistics or the model, and the statistics know nothing about
the service.

**Why the API is split from the UI.** Prediction is a service, not a button. The
separation makes two things true that a single Streamlit script could not: the
prediction endpoint can be integrated by something other than this UI, and the
UI provably contains no business logic — it imports no `rwsat` internals and
talks to the API over HTTP only (enforced by a test). The UI builds its input
form dynamically from `GET /schema`; hardcoding fields in the frontend would be
a bug. The service loads its artifacts once at startup and never runs a
permutation test or a training pass inside a request.

---

## Methodology

**One statistic for every factor.** The 18 features are a mix of nominal,
ordinal and numeric, and their textbook effect sizes (Cramér's V, eta squared,
Spearman's rho) are not comparable with one another. So every factor is scored
the same way: **`delta_logloss`**, the out-of-fold log-loss improvement of a
univariate model over the prior baseline (the class frequencies). It is the sole
basis for ranking and verdicts; the native effect sizes are reported alongside
as descriptive columns only (D-002). The univariate model encodes every feature
as one-hot over its levels — numeric features quantile-binned — so a
**non-monotone** association is caught rather than mistaken for noise (D-006).
`delta_logloss` can be slightly negative for a useless feature; that is expected,
since out-of-fold log loss penalises unhelpful degrees of freedom.

**Accounting for chance.** For each factor the target is shuffled 2,000 times and
the score recomputed, building a null distribution of what pure chance produces
on this exact data. The observed score is placed on it; the share of shuffles
doing at least as well is the p-value, under the add-one convention
`(1 + k) / (1 + N)` so it can never be exactly zero (D-008). All p-values are
Benjamini–Hochberg adjusted across the full family of 18 features before any
verdict is issued — uncorrected, chance alone would hand roughly one of them a
"significant" result.

**How to read a verdict.**
- **signal** — beats chance after correction *and* the confidence interval
  clears the negligibility line. Worth acting on.
- **noise** — chance regularly does at least as well, and the interval rules out
  anything but a negligible improvement.
- **inconclusive** — neither can be established. An honest state, never collapsed
  into the other two.

**Why log loss, not accuracy.** Log loss is a proper scoring rule, and the
product returns probabilities, so it is the primary metric for model selection
and the permutation tests. Balanced accuracy is reported as secondary. "Beats
baseline" means beating the `DummyClassifier(strategy="prior")` by more than two
standard deviations of the *paired per-fold* differences on the shared CV folds,
not either model's own spread (D-009). Model selection then picks the simplest
model that is statistically indistinguishable from the best.

**Permutation importance describes the model, not the data.** The model card
reports permutation importance (with confidence intervals; tree
`feature_importances_` are never used). These describe the fitted model's
behaviour and **can be positive for a model that itself loses to the baseline** —
in this run `Years_of_Experience` ranks first by both fitted models while the
factor screen calls it noise. Factor evidence therefore lives only on the Factor
Explorer screen; the model card fences its importances behind that caveat.

The held-out test set is measured exactly once, after all model choices are made
on cross-validation.

---

## Decisions and tradeoffs

Every non-trivial choice is recorded in [`DECISIONS.md`](DECISIONS.md) with its
context, the alternatives, and — required for every entry — its **cost**.
Fourteen decisions are logged (D-000 … D-018), including: specification before
code, the product framing that follows the signal check, the single common
statistic, treating the literal string `"None"` as a category rather than
missing data, keeping onsite employees in the sample, the add-one permutation
convention, and shipping the precomputed artifacts to get a fast cold start.

---

## Limitations

One limitation is the inclusion of onsite employees in the dataset. There are 1637 onsite employees out of 5000 rows, while the target is described as satisfaction with remote work. For these employees, the interpretation of the target is therefore weaker. However, this was an intentional decision. Removing approximately one third of the dataset would also remove the only meaningful comparison involving `Work_Location`. The analysis therefore keeps these records and interprets the target as satisfaction with the employee's current work format in general.

Another limitation is that the screening method checks one factor at a time. By construction, it cannot detect pairwise effects where two factors only become relevant when they occur together. The model-based permutation test provides a separate check for this possibility. If an important interaction existed, a flexible model should be able to use it and perform better than the baseline. This did not happen. Therefore, the difference between the screening result and the model result would have been a finding rather than a contradiction.

The third limitation concerns features that are close to the decision threshold. The overall conclusion for the full dataset is stable: both tested seeds produce the same "no signal" conclusion. However, the classification of individual features close to the threshold can change between "noise" and "inconclusive" when the seed is changed. This is a property of the threshold itself rather than a defect in the method. The boundary for a "signal" classification did not change in either run.

Finally, the project does not include a database, authentication, or cloud deployment. These components are outside the scope because this is an evaluation project rather than a SaaS product. A database would become justified if the data arrived in repeated time-based batches or if predictions had to be stored for later auditing. Neither requirement exists in the current project.

---

## How the AI tooling was used

This repository was built with an AI coding assistant under a project
configuration (a `CLAUDE.md`, three specialised subagents, two rule sets and one
project skill). Where the assistant was steered, corrected, or pushed past its
first answer is documented concretely in
[`docs/ai-collaboration.md`](docs/ai-collaboration.md).

---

## Repo map

| Path | What it is |
|---|---|
| `src/rwsat/config.py` | every seed, path and threshold — the single source of constants |
| `src/rwsat/data.py` | schema, `load_raw`/`validate`/`prepare`/`split` (M1) |
| `src/rwsat/features.py` | the sklearn preprocessor and form metadata (M2) |
| `src/rwsat/stats.py` | the common statistic, permutation nulls, verdicts (M3) |
| `src/rwsat/model.py` | training, calibration, model cards, prediction (M4) |
| `src/rwsat/api.py` | FastAPI service (M5) |
| `src/rwsat/cli.py` | `validate` / `verify-signal` / `train` (M7) |
| `app/main.py` | Streamlit UI — HTTP only (M6) |
| `tests/` | 59 tests: leakage, synthetic known-answer, edge cases, endpoints, UI degradation |
| `SPEC.md` | the module specification the code is built against |
| `DECISIONS.md` | the decision log (D-000 … D-013), each with a cost |
| `docs/ai-collaboration.md` | how the assistant was used and overridden |
| `data/raw/LICENSE.md` | dataset source and license |

## Commands

```bash
make install        # uv sync
make test           # uv run pytest -q   (59 tests)
make lint           # ruff check + mypy (strict)
make train          # regenerate the model artifact
make run            # docker compose up
```

## Dataset license

Remote Work & Mental Health, by *waqi786* on Kaggle, published under the **MIT**
license. The CSV is committed unmodified so the product runs on a clean machine
with no Kaggle credentials. Full provenance is in
[`data/raw/LICENSE.md`](data/raw/LICENSE.md). The dataset is synthetic; its known
internal inconsistencies are part of this project's findings (see the Headline
finding above and the Data quality section of `SPEC.md`), not defects introduced
here.
