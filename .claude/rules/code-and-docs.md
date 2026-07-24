---
description: Code structure and documentation discipline
paths:
  - "src/**/*.py"
  - "app/**/*.py"
  - "tests/**/*.py"
  - "*.md"
  - "docs/**/*.md"
---

# Code and documentation standards

The evaluation criterion is: would another engineer be able to pick this up.

## Structure
- One module, one layer. A module importing both the model and the UI is a design
  error.
- Type annotations on all public functions; `mypy` passes.
- Structured results are dataclasses or pydantic models, never bare dicts or tuples.
- Paths are `pathlib.Path`.
- Functions longer than 50 lines get split. No module-level computation on import.

## Errors
- No bare `except:`. Catch the specific exception.
- Error messages say what to do, not only what broke.
- Return a report where the caller can continue; raise where it cannot.

## Tests
- `stats.py` and `features.py` are mandatory to cover: these are the places where
  a bug does not crash, it quietly distorts the conclusion.
- Synthetic-data tests with a known answer matter more than tests on real data.
- A leakage test is required.
- Tests do not hit the network and do not require a trained model unless marked.

## Forbidden
- Commented-out code, `print` in library code, magic numbers outside `config.py`,
  `TODO` and `pass` placeholders in committed code.

## When a DECISIONS.md entry is required
- One of several workable options was chosen
- An obvious or conventional approach was rejected
- A scope limit was accepted
- A property of the data changed the architecture

Format: context, options, decision, rationale, **cost**, status. An entry without
a cost is incomplete — every decision has one, and not seeing it means the
alternatives were not understood.

## When a docs/ai-collaboration.md entry is required
- The assistant's suggestion was rejected or substantially rewritten
- The assistant was steered in a non-obvious way
- The assistant made a mistake a human caught
- The assistant was pushed past the first working answer

Format: task, what the assistant proposed, why it did not fit, what was done
instead, what it says. "Used Claude to generate code" is not an entry.

## Forbidden in prose
- Evaluative adjectives without a number behind them: powerful, advanced, robust
- "The model performs well" without a baseline comparison
- Any driver claim without a reference to a concrete `EffectResult`

README run instructions are written only after being executed literally on a
clean environment. Steps reconstructed from memory are not acceptable.
