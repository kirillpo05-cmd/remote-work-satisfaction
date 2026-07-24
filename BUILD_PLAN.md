# Build plan

Phases with gates, not a calendar. Each gate is a point where you stop, look at
what came out, and decide — that is where the judgement the brief asks about
actually happens. Deleting this file before submission is fine; it is internal.

Rough wall-clock estimates assume an uninterrupted session with an AI assistant.

---

## Phase 0 — Scaffold and the signal gate  (~2h)

- Repo init, `uv init`, directory structure, ruff/mypy/pytest config
- `src/rwsat/config.py`: seed, paths, thresholds
- M1 data layer: schema, validation, split, with edge-case tests
- Reconcile the expected schema in SPEC.md against the actual CSV
- Run the `verify-signal` skill in full

**GATE 1 — human decision, do not delegate.**
Read the verdict. Choose the product framing. Write D-001 in your own words,
including its cost. Everything downstream depends on this.

Do not build the model, API or UI before this gate.

---

## Phase 1 — Inference layer  (~2h)
Agent: `data-scientist`

- M3 `stats.py`: effect sizes by variable type, bootstrap CIs, BH correction
- `permutation_null` and the three-level verdict with plain-English explanations
- Synthetic tests with known answers: random target -> `noise`, injected
  association -> `signal`
- Cache the `SignalReport` to `artifacts/`

**GATE 2.** Rerun the permutation test with a different seed. If the verdict
moves, the test is implemented wrong. Fix it before building on top of it.

---

## Phase 2 — Model layer  (~2h)
Agent: `data-scientist`

- M2 `features.py`: Pipeline, feature metadata for the form, leakage test
- M4 `model.py`: three models, CV, single test measurement, model permutation test
- Calibration, Brier score, proportional-odds check
- `ModelCard`, selection rule, serialisation

Hard limit: exactly three models. Hyperparameter tuning is out of scope.

**GATE 3.** Do the permutation importances from the model and the effect sizes
from `stats.py` tell the same story? If not, find out why before proceeding —
that discrepancy is either a bug or a finding, and both belong in the docs.

---

## Phase 3 — Service and interface  (~2-3h)
Agent: `app-engineer`

- M5 endpoints with `TestClient` coverage including error branches
- M6 five screens, form generated from `/schema`, four states each
- The permutation null histogram — the product's key visual

---

## Phase 4 — Packaging  (~1h)
Agent: `app-engineer`

- Dockerfile, compose.yaml with healthcheck, Makefile, `.env.example`
- Dataset committed with `LICENSE.md` naming the source
- **Clean-machine check:** fresh clone, `docker compose up`, zero extra steps
- Docker-free path verified the same way
- Full `reviewer` pass; clear all BLOCKER findings

---

## Phase 5 — Documentation  (~2h)
Human-led. This is not leftover time.

- README: what it is, the headline finding first, quick start, architecture and
  why it is split this way, methodology, decisions and tradeoffs, limitations,
  how the AI tooling was used, repo map
- DECISIONS.md: at least eight entries, each with a cost
- docs/ai-collaboration.md: at least five concrete episodes
- Re-run the README instructions literally, as written

**GATE 4.** Write the rationale sentences yourself. A decision log in fluent
generated prose with no rough edges reads as fabricated and undermines the exact
criterion it is meant to satisfy.

---

## Phase 6 — Optional and submission  (~1h)

- M8 agent layer, only if everything above is complete
- Final `reviewer` pass, clear BLOCKER and MAJOR
- Meaningful commit history, not one squashed dump
- Private repo, access granted to `linards-kalvans`
- Final check: clone into a new directory and run from the README

---

## Cutting order under pressure

Cut in this order: M8 agent layer, then the Model Card screen down to key
metrics, then the gradient boosting model (leaving dummy and ordinal logit).

Never cut: leakage tests, `stats.py` tests, one-command startup, README and
DECISIONS.md.
