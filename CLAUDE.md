# CLAUDE.md — RWSAT (Remote Work Satisfaction Explorer)

## What this is
A data product built on the Kaggle dataset `waqi786/remote-work-and-mental-health`
(5,000 rows, 20 columns). It answers two questions:
1. Which factors drive employee satisfaction with remote work?
2. Given an employee's attributes, what satisfaction level do we predict?

## Guiding principle
**Honesty over accuracy.** The product must separate signal from noise and say so
plainly when a conclusion is not statistically supported. Every effect and every
prediction ships with a measure of uncertainty. Predictive accuracy is explicitly
not a success criterion here; defensible reasoning is.

## Stack
- Python 3.12, `uv` for dependencies and execution
- pandas, scikit-learn, statsmodels (ordinal regression), scipy
- FastAPI for the service layer, Streamlit for the UI
- pytest, ruff, mypy
- Docker + Compose for one-command startup

## Architecture
```
data/raw/*.csv  ->  src/rwsat/data.py      schema, validation, split
                ->  src/rwsat/features.py  sklearn Pipeline, no leakage
                ->  src/rwsat/stats.py     effect sizes, corrections, permutation tests
                ->  src/rwsat/model.py     training, calibration, evaluation
                ->  src/rwsat/api.py       FastAPI
                ->  app/main.py            Streamlit, talks to the API only
```

## Rules
1. No code without a matching section in `SPEC.md`. If the spec is thin, extend
   the spec first — never guess in code.
2. Train/test split happens **before** any preprocessing. All preprocessing lives
   inside `sklearn.Pipeline`. Fitting on test data is a critical bug.
3. Tree-based `feature_importances_` are never used as evidence of influence.
   Permutation importance with confidence intervals only.
4. Every claim about a factor carries an effect size, a confidence interval, and
   a multiple-comparison correction.
5. All randomness draws its seed from `src/rwsat/config.py`. Reruns must be identical.
6. The application starts and works with no API keys and no environment variables.
7. No TODO stubs, no `pass` placeholders, no commented-out code in commits.
8. Whenever a choice had a viable alternative, record it in `DECISIONS.md`.
   Whenever the assistant's suggestion was rejected or rewritten, record it in
   `docs/ai-collaboration.md`. Both files are graded deliverables.

## Deliberately out of scope
Database, authentication, multi-user support, cloud deployment, online retraining,
hyperparameter tuning. This is a scoped evaluation project, not a SaaS.

## Commands
```bash
uv sync
uv run pytest -q
uv run ruff check src tests && uv run mypy src
uv run python -m rwsat.cli verify-signal
uv run python -m rwsat.cli train
docker compose up            # API on :8000, UI on :8501
```

## Documents
- `SPEC.md` — module specifications: stories, data, API, screens, logic, edge cases
- `BUILD_PLAN.md` — build phases and gates (internal, delete before submission)
- `DECISIONS.md` — decision log
- `docs/ai-collaboration.md` — record of how the AI assistant was used and overridden
- `README.md` — written last, by hand, from the two logs above

## The Phase 0 gate
The first task in this project is the `verify-signal` skill. Its outcome decides
the product framing. Do not build the model, API or UI layers before it completes.
- Signal found: emphasise driver explanation and prediction quality.
- No signal found: emphasise proving its absence and protecting the user from a
  false conclusion. This is a valid result, not a failure, and the product is
  designed to remain useful in this case.
