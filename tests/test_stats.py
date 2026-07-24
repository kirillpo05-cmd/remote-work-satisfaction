"""Synthetic known-answer tests for M3 (SPEC M3). Written before the implementation.

The central fixture asserts both directions of the non-monotone requirement
(BUILD_PLAN Phase 1): a U-shaped association on a numeric feature yields
`signal`, and a pure-noise feature with the same number of levels yields
`noise`. Without the second assertion an over-eager screen would pass.
"""

import numpy as np
import pandas as pd
import pytest

from rwsat.config import ALPHA, DELTA_NEGLIGIBLE
from rwsat.stats import SignalReport, effect_for, permutation_null, signal_report

TARGET = "target"
CLASSES = ["Unsatisfied", "Neutral", "Satisfied"]


def make_target(codes: np.ndarray) -> pd.Categorical:
    return pd.Categorical.from_codes(codes, categories=CLASSES, ordered=True)


def u_and_noise_frame(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """One numeric feature with a U-shaped association, one pure-noise numeric
    feature. Both are quantile-binned to the same number of levels by the
    screen, so the only difference between them is the injected association."""
    rng = np.random.default_rng(seed)
    x = rng.integers(20, 61, size=n).astype(float)
    extreme = np.abs(x - 40) > 12
    y = np.where(
        extreme,
        rng.choice(3, size=n, p=[0.5, 0.3, 0.2]),
        rng.choice(3, size=n, p=[0.2, 0.3, 0.5]),
    )
    noise = rng.integers(20, 61, size=n).astype(float)
    return pd.DataFrame({"u_feature": x, "noise_feature": noise, TARGET: make_target(y)})


@pytest.fixture(scope="module")
def u_and_noise_report() -> SignalReport:
    return signal_report(u_and_noise_frame(), TARGET, n_boot=300, n_perm=200, seed=11)


# --- the non-monotone fixture, both directions ---------------------------


def test_u_shape_yields_signal(u_and_noise_report: SignalReport) -> None:
    effect = {e.feature: e for e in u_and_noise_report.effects}["u_feature"]
    assert effect.verdict == "signal"
    assert effect.delta_logloss > DELTA_NEGLIGIBLE
    assert effect.delta_ci_low > DELTA_NEGLIGIBLE
    assert effect.permutation_p_adj < ALPHA


def test_pure_noise_same_levels_yields_noise(u_and_noise_report: SignalReport) -> None:
    effect = {e.feature: e for e in u_and_noise_report.effects}["noise_feature"]
    assert effect.verdict == "noise"
    assert effect.delta_ci_high < DELTA_NEGLIGIBLE


def test_dataset_verdict_follows_the_common_statistic(
    u_and_noise_report: SignalReport,
) -> None:
    assert u_and_noise_report.n_signal == 1
    assert u_and_noise_report.dataset_verdict == "signal_present"
    assert u_and_noise_report.n_features_tested == 2


# --- random target -> no detectable signal --------------------------------


def test_random_target_yields_no_signal() -> None:
    rng = np.random.default_rng(1)
    n = 900
    df = pd.DataFrame(
        {
            "numeric": rng.normal(size=n),
            "nominal": pd.Categorical(rng.choice(list("ABCD"), size=n)),
            "ordered": pd.Categorical(
                rng.choice(["Low", "Medium", "High"], size=n),
                categories=["Low", "Medium", "High"],
                ordered=True,
            ),
            TARGET: make_target(rng.integers(0, 3, size=n)),
        }
    )
    report = signal_report(df, TARGET, n_boot=200, n_perm=100, seed=5)
    assert report.dataset_verdict == "no_detectable_signal"
    assert all(e.verdict in ("noise", "inconclusive") for e in report.effects)


# --- statistical mechanics -------------------------------------------------


def test_permutation_p_never_zero_and_bh_never_shrinks(
    u_and_noise_report: SignalReport,
) -> None:
    for e in u_and_noise_report.effects:
        assert e.permutation_p >= 1 / 201  # add-one floor at n_perm=200
        assert e.permutation_p_adj >= e.permutation_p
        assert e.permutation_p_adj <= 1.0


def test_effects_sorted_by_delta_descending(u_and_noise_report: SignalReport) -> None:
    deltas = [e.delta_logloss for e in u_and_noise_report.effects]
    assert deltas == sorted(deltas, reverse=True)


def test_permutation_null_is_seeded_and_sized() -> None:
    df = u_and_noise_frame(n=400, seed=2)
    null_a = permutation_null(df, "noise_feature", TARGET, n_perm=30, seed=9)
    null_b = permutation_null(df, "noise_feature", TARGET, n_perm=30, seed=9)
    assert null_a.shape == (30,)
    assert np.array_equal(null_a, null_b)


def test_effect_for_matches_signal_report_entry() -> None:
    df = u_and_noise_frame(n=400, seed=3)
    single = effect_for(df, "u_feature", TARGET, n_boot=100, n_perm=50, seed=7)
    family = signal_report(df, TARGET, n_boot=100, n_perm=50, seed=7)
    from_report = {e.feature: e for e in family.effects}["u_feature"]
    assert single == from_report


# --- edge cases ------------------------------------------------------------


def test_constant_feature_is_noise_and_says_so() -> None:
    rng = np.random.default_rng(4)
    n = 300
    df = pd.DataFrame(
        {
            "constant": np.ones(n),
            "filler": rng.normal(size=n),
            TARGET: make_target(rng.integers(0, 3, size=n)),
        }
    )
    report = signal_report(df, TARGET, n_boot=50, n_perm=20, seed=13)
    effect = {e.feature: e for e in report.effects}["constant"]
    assert effect.verdict == "noise"
    assert effect.delta_logloss == 0.0
    assert "constant" in effect.explanation


def test_rare_categories_are_merged() -> None:
    rng = np.random.default_rng(6)
    n = 300
    values = np.array(["common_a", "common_b"])[rng.integers(0, 2, size=n)]
    values[:3] = "rare"  # 3 < MIN_CATEGORY_N
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(values),
            TARGET: make_target(rng.integers(0, 3, size=n)),
        }
    )
    report = signal_report(df, TARGET, n_boot=50, n_perm=20, seed=13)
    effect = {e.feature: e for e in report.effects}["cat"]
    assert "merged" in effect.explanation


def test_object_dtype_feature_is_rejected_with_prepare_hint() -> None:
    rng = np.random.default_rng(10)
    n = 120
    df = pd.DataFrame(
        {
            "untyped": rng.choice(["Low", "Medium", "High"], size=n),  # object dtype
            TARGET: make_target(rng.integers(0, 3, size=n)),
        }
    )
    with pytest.raises(ValueError, match="prepare"):
        signal_report(df, TARGET, n_boot=10, n_perm=5, seed=1)


def test_nan_rows_are_excluded_and_n_reflects_it() -> None:
    df = u_and_noise_frame(n=400, seed=8)
    df.loc[df.index[:25], "noise_feature"] = np.nan
    report = signal_report(df, TARGET, n_boot=50, n_perm=20, seed=13)
    by_name = {e.feature: e for e in report.effects}
    assert by_name["noise_feature"].n == 375
    assert by_name["u_feature"].n == 400
