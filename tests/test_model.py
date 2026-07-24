"""Tests for M4 (SPEC M4). Written before the implementation.

Two synthetic frames with known answers drive everything: one with a random
target (no model may beat baseline, training must still succeed) and one with
a strong Stress_Level -> satisfaction association (a real model must win).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rwsat.config import SEED
from rwsat.data import TARGET, prepare
from rwsat.model import (
    ModelCard,
    TrainResult,
    load,
    predict_one,
    save,
    select_model,
    train_all,
)
from test_data import synthetic_frame

CLASSES = ["Unsatisfied", "Neutral", "Satisfied"]


def _decorrelate(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # synthetic_frame cycles values, which makes whole columns copies of each
    # other (a singular design). Shuffling each column independently removes
    # the collinearity without changing any marginal distribution.
    for column in df.columns:
        if column not in ("Employee_ID", TARGET):
            df[column] = rng.permutation(df[column].to_numpy())
    return df


def random_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _decorrelate(synthetic_frame(n), rng)
    df[TARGET] = np.array(CLASSES)[rng.integers(0, 3, size=n)]
    return prepare(df)


def signal_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = _decorrelate(synthetic_frame(n), rng)
    mapping = {"Low": "Satisfied", "Medium": "Neutral", "High": "Unsatisfied"}
    determined = df["Stress_Level"].map(mapping).to_numpy()
    noise = np.array(CLASSES)[rng.integers(0, 3, size=n)]
    df[TARGET] = np.where(rng.random(n) < 0.75, determined, noise)
    return prepare(df)


@pytest.fixture(scope="module")
def random_result() -> TrainResult:
    return train_all(random_frame(), n_perm_model=5, seed=SEED)


@pytest.fixture(scope="module")
def signal_result() -> TrainResult:
    return train_all(signal_frame(), n_perm_model=5, seed=SEED)


# --- structure -------------------------------------------------------------


def test_three_primary_cards_present(random_result: TrainResult) -> None:
    assert {"dummy", "ordinal_logit", "gradient_boosting"} <= set(random_result.cards)
    for card in random_result.cards.values():
        assert {"log_loss", "balanced_accuracy", "accuracy", "macro_f1", "brier"} <= set(
            card.metrics
        )
        assert card.baseline_metrics  # every metric sits next to baseline
        assert len(card.cv_fold_logloss) == 5
        assert card.n_train + card.n_test == 600


def test_calibration_per_class_and_brier(random_result: TrainResult) -> None:
    card = random_result.cards["gradient_boosting"]
    assert set(card.calibration) == set(CLASSES)
    for points in card.calibration.values():
        assert all(len(point) == 2 for point in points)
    assert 0.0 <= card.metrics["brier"] <= 2.0


def test_permutation_importance_has_cis(random_result: TrainResult) -> None:
    effects = random_result.cards["gradient_boosting"].feature_effects
    assert len(effects) == 18
    for entry in effects:
        assert {"feature", "importance_mean", "ci_low", "ci_high"} <= set(entry)
        assert entry["ci_low"] <= entry["importance_mean"] <= entry["ci_high"]


def test_proportional_odds_check_recorded(random_result: TrainResult) -> None:
    assert random_result.cards["ordinal_logit"].proportional_odds_ok is not None


# --- negative result: training succeeds, nothing pretends to win ------------


def test_random_target_no_model_beats_baseline(random_result: TrainResult) -> None:
    for name in ("ordinal_logit", "gradient_boosting"):
        card = random_result.cards[name]
        assert card.is_better_than_baseline is False
        assert card.warnings  # explicit warning enters the card
    assert random_result.selected == "dummy"


def test_prediction_on_random_data_is_honest(random_result: TrainResult) -> None:
    prediction = predict_one(random_result.model, {})
    assert prediction.confidence == "not_better_than_guessing"
    assert prediction.warning
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert len([f for f in prediction.flags if f.startswith("imputed:")]) == 18
    for cls, p in prediction.probabilities.items():
        assert p == pytest.approx(prediction.prior_probabilities[cls], abs=1e-9)


# --- positive control: injected signal is found -----------------------------


def test_injected_signal_beats_baseline_and_moves_predictions(
    signal_result: TrainResult,
) -> None:
    assert any(
        signal_result.cards[name].is_better_than_baseline
        for name in ("ordinal_logit", "gradient_boosting")
    )
    assert signal_result.selected != "dummy"

    low = predict_one(signal_result.model, {"Stress_Level": "Low"})
    high = predict_one(signal_result.model, {"Stress_Level": "High"})
    assert low.probabilities["Satisfied"] > high.probabilities["Satisfied"]
    assert low.confidence in ("high", "low")


# --- prediction flags --------------------------------------------------------


def test_predict_flags(signal_result: TrainResult) -> None:
    model = signal_result.model
    assert "out_of_range" in predict_one(model, {"Age": 200}).flags
    assert "unknown_category" in predict_one(model, {"Gender": "Alien"}).flags
    missing_age = predict_one(model, {"Stress_Level": "Low"})
    assert "imputed:Age" in missing_age.flags
    with pytest.raises(ValueError, match="Employee_ID"):
        predict_one(model, {"Employee_ID": "EMP0001"})


# --- persistence -------------------------------------------------------------


def test_save_load_roundtrip(signal_result: TrainResult, tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    save(signal_result.model, path)
    loaded = load(path)
    payload = {"Stress_Level": "High", "Age": 30}
    assert predict_one(loaded, payload).probabilities == pytest.approx(
        predict_one(signal_result.model, payload).probabilities
    )


# --- the selection rule in isolation ----------------------------------------


def _card_stub(name: str, fold_logloss: list[float]) -> ModelCard:
    return ModelCard(
        model_name=name,
        trained_at=None,  # type: ignore[arg-type]
        n_train=0,
        n_test=0,
        metrics={},
        baseline_metrics={},
        lift_over_baseline={},
        cv_mean=float(np.mean(fold_logloss)),
        cv_std=float(np.std(fold_logloss)),
        permutation_test_p=1.0,
        is_better_than_baseline=False,
        cv_fold_logloss=fold_logloss,
        calibration={},
        proportional_odds_ok=None,
        feature_effects=[],
        warnings=[],
    )


def test_select_model_prefers_simplest_on_a_tie() -> None:
    cards = {
        "dummy": _card_stub("dummy", [1.0, 1.0, 1.0, 1.0, 1.0]),
        "ordinal_logit": _card_stub("ordinal_logit", [1.0, 1.0, 1.0, 1.0, 1.0]),
        "gradient_boosting": _card_stub("gradient_boosting", [1.001, 0.999, 1.0, 1.0, 1.0]),
    }
    assert select_model(cards) == "dummy"


def test_select_model_picks_a_clear_winner() -> None:
    cards = {
        "dummy": _card_stub("dummy", [1.0, 1.0, 1.0, 1.0, 1.0]),
        "ordinal_logit": _card_stub("ordinal_logit", [0.5, 0.51, 0.49, 0.5, 0.5]),
        "gradient_boosting": _card_stub("gradient_boosting", [0.6, 0.61, 0.59, 0.6, 0.6]),
    }
    assert select_model(cards) == "ordinal_logit"
