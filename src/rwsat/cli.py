"""M7 CLI — the three commands quoted in CLAUDE.md (SPEC M7).

- validate       print the ValidationReport for the raw CSV; exit 1 when invalid
- verify-signal  screen the training split, write artifacts/signal_report.json
- train          train, select, serialise the artefact and the ModelCards
"""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any

from rwsat.config import ARTIFACTS_DIR, MODEL_CARDS_PATH, MODEL_PATH, SIGNAL_REPORT_PATH
from rwsat.data import TARGET, load_raw, prepare, split, validate
from rwsat.model import ModelCard, save, train_all
from rwsat.stats import signal_report


def _cmd_validate() -> int:
    report = validate(load_raw())
    print(json.dumps(asdict(report), indent=2))
    if not report.is_valid:
        print("Validation FAILED — see the report above.")
        return 1
    print("Validation passed.")
    return 0


def _cmd_verify_signal() -> int:
    df = prepare(load_raw())
    train_df, _test_df = split(df)  # the held-out split never enters screening
    train_df = train_df.reset_index(drop=True)
    report = signal_report(train_df, TARGET)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    SIGNAL_REPORT_PATH.write_text(json.dumps(asdict(report), indent=2))
    strongest = report.effects[0]
    print(
        f"Dataset verdict: {report.dataset_verdict}. "
        f"{report.n_signal} of {report.n_features_tested} features earn a `signal` "
        f"verdict under the common statistic — the out-of-fold log-loss improvement "
        f"of a univariate model over the prior baseline, each placed on its own "
        f"permutation null. {report.family_wise_note} "
        f"The strongest candidate is {strongest.feature} "
        f"(delta={strongest.delta_logloss:+.5f}, adjusted p={strongest.permutation_p_adj:.3f}, "
        f"verdict: {strongest.verdict}). {strongest.explanation}"
    )
    print(f"Report written to {SIGNAL_REPORT_PATH}")
    return 0


def _card_as_dict(card: ModelCard) -> dict[str, Any]:
    payload = asdict(card)
    payload["trained_at"] = card.trained_at.isoformat()
    return payload


def _cmd_train() -> int:
    df = prepare(load_raw())
    result = train_all(df)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    save(result.model, MODEL_PATH)
    MODEL_CARDS_PATH.write_text(
        json.dumps(
            {
                "selected": result.selected,
                "cards": {name: _card_as_dict(card) for name, card in result.cards.items()},
            },
            indent=2,
        )
    )
    print(f"Selected model: {result.selected}")
    for name, card in result.cards.items():
        print(
            f"  {name:<18s} cv_logloss={card.cv_mean:.4f}±{card.cv_std:.4f}  "
            f"test_logloss={card.metrics['log_loss']:.4f}  "
            f"beats_baseline={card.is_better_than_baseline}"
        )
    print(f"Artefacts written: {MODEL_PATH}, {MODEL_CARDS_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rwsat.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="print the ValidationReport; exit 1 when invalid")
    sub.add_parser("verify-signal", help="screen the training split, cache the SignalReport")
    sub.add_parser("train", help="train, select, serialise the model and its cards")
    args = parser.parse_args(argv)
    if args.command == "validate":
        return _cmd_validate()
    if args.command == "verify-signal":
        return _cmd_verify_signal()
    return _cmd_train()


if __name__ == "__main__":
    sys.exit(main())
