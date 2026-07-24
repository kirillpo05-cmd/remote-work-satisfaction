"""M3 — inference layer: the common statistic, permutation nulls, verdicts (SPEC M3).

One statistic is computed for every feature regardless of type:
``delta_logloss``, the out-of-fold log-loss improvement of a univariate model
over the prior baseline. It is the sole basis for ranking and verdicts; native
effect sizes (Cramer's V, eta squared, Spearman's rho) are descriptive only.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

from rwsat.config import (
    ALPHA,
    DELTA_NEGLIGIBLE,
    MIN_CATEGORY_N,
    N_BOOT,
    N_PERM,
    N_SPLITS,
    N_UNIVARIATE_BINS,
    NOISE_PERMUTATION_P,
    SEED,
)

logger = logging.getLogger(__name__)

_EPS = 1e-15

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

Verdict = Literal["signal", "inconclusive", "noise"]
NativeTest = Literal["chi2", "kruskal", "spearman"]


@dataclass(frozen=True)
class EffectResult:
    feature: str
    # --- the common statistic: sole basis for ranking and verdict ---
    delta_logloss: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p: float
    permutation_p_adj: float
    null_mean: float
    null_p95: float
    verdict: Verdict
    explanation: str
    n: int
    # --- native effect size: descriptive only ---
    native_test: NativeTest
    native_effect_size: float
    native_effect_name: str


@dataclass(frozen=True)
class SignalReport:
    effects: list[EffectResult]
    n_features_tested: int
    n_signal: int
    family_wise_note: str
    dataset_verdict: Literal["signal_present", "no_detectable_signal"]


@dataclass(frozen=True)
class _Encoded:
    """A feature prepared for the univariate screen."""

    kind: Literal["numeric", "ordinal", "categorical"]
    values: FloatArray | None  # numeric: raw values, binned inside each fold
    codes: IntArray | None  # non-numeric: fixed level codes
    n_levels: int
    n_merged: int  # rare levels folded into one bucket (MIN_CATEGORY_N)
    mask: BoolArray  # rows with a usable feature value


def _encode(series: pd.Series) -> _Encoded:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(dtype=float)
        return _Encoded(
            kind="numeric",
            values=values,
            codes=None,
            n_levels=N_UNIVARIATE_BINS,
            n_merged=0,
            mask=~np.isnan(values),
        )
    if not isinstance(series.dtype, pd.CategoricalDtype):
        # Typing follows dtypes, and dtypes come from prepare() (D-011).
        # Guessing on raw strings would silently treat ordinals as nominal.
        raise ValueError(
            f"Feature {series.name!r} has dtype {series.dtype}, which carries no "
            f"type information. Run rwsat.data.prepare() on the frame first — it "
            f"assigns numeric/ordinal/categorical dtypes from SCHEMA."
        )
    categorical = series.cat
    kind: Literal["ordinal", "categorical"] = (
        "ordinal" if series.dtype.ordered else "categorical"
    )
    codes = np.asarray(categorical.codes, dtype=np.int64)
    n_levels = len(categorical.categories)
    mask = codes >= 0  # pandas encodes missing as -1

    counts = np.bincount(codes[mask], minlength=n_levels)
    rare = np.flatnonzero((counts > 0) & (counts < MIN_CATEGORY_N))
    n_merged = 0
    if len(rare) > 0:
        # All rare levels pool into one "Other" bucket (SPEC M3 edge case).
        # With a single rare level the pooling is a numeric no-op, but the
        # merge is still recorded so the instability is visible.
        bucket = int(rare[0])
        codes = np.where(np.isin(codes, rare), bucket, codes)
        n_merged = len(rare)
    return _Encoded(
        kind=kind, values=None, codes=codes, n_levels=n_levels, n_merged=n_merged, mask=mask
    )


def _bin_codes(train_values: FloatArray, all_values: FloatArray) -> IntArray:
    """Quantile-bin using edges fitted on the fold-train values only."""
    quantiles = np.linspace(0, 1, N_UNIVARIATE_BINS + 1)[1:-1]
    edges = np.quantile(train_values, quantiles)
    return np.asarray(np.searchsorted(edges, all_values, side="right"), dtype=np.int64)


def _oof_row_deltas(encoded: _Encoded, y: IntArray, seed: int) -> FloatArray:
    """Per-row (baseline log loss - univariate model log loss), out of fold."""
    n = len(y)
    n_classes = int(y.max()) + 1
    base_ll = np.empty(n)
    model_ll = np.empty(n)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, va in skf.split(np.zeros(n), y):
        if encoded.values is not None:
            values = encoded.values
            codes = _bin_codes(values[tr], values)
        else:
            assert encoded.codes is not None
            codes = encoded.codes
        eye = np.eye(encoded.n_levels)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(eye[codes[tr]], y[tr])
        proba = np.asarray(clf.predict_proba(eye[codes[va]]), dtype=float)
        freqs = np.bincount(y[tr], minlength=n_classes) / len(tr)
        model_ll[va] = -np.log(np.clip(proba[np.arange(len(va)), y[va]], _EPS, None))
        base_ll[va] = -np.log(np.clip(freqs[y[va]], _EPS, None))
    return np.asarray(base_ll - model_ll, dtype=np.float64)


def _native_effect(series: pd.Series, y: IntArray) -> tuple[NativeTest, float, str]:
    n = len(y)
    if pd.api.types.is_numeric_dtype(series):
        x = series.to_numpy(dtype=float)
        n_classes = int(y.max()) + 1
        groups = [x[y == c] for c in range(n_classes)]
        h_stat, _ = sps.kruskal(*groups)
        eta_sq = max(0.0, (float(h_stat) - n_classes + 1) / (n - n_classes))
        return "kruskal", eta_sq, "eta_squared"
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        rho, _ = sps.spearmanr(np.asarray(series.cat.codes, dtype=float), y)
        return "spearman", float(rho), "spearman_rho"
    table = pd.crosstab(series, pd.Series(y)).to_numpy()
    chi2, _, _, _ = sps.chi2_contingency(table)
    v = float(np.sqrt(float(chi2) / (n * (min(table.shape) - 1))))
    return "chi2", v, "cramers_v"


def _feature_list(df: pd.DataFrame, target: str, features: list[str] | None) -> list[str]:
    if features is not None:
        return features
    return [str(c) for c in df.columns if c != target]


def _target_codes(df: pd.DataFrame, target: str) -> IntArray:
    series = df[target]
    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")
    codes = np.asarray(series.cat.codes, dtype=np.int64)
    if (codes < 0).any():
        raise ValueError(f"Target {target} contains missing values; drop them before screening.")
    return codes


def _verdict(perm_p_adj: float, perm_p: float, ci_low: float, ci_high: float) -> Verdict:
    if perm_p_adj < ALPHA and ci_low > DELTA_NEGLIGIBLE:
        return "signal"
    if perm_p > NOISE_PERMUTATION_P and ci_high < DELTA_NEGLIGIBLE:
        return "noise"
    return "inconclusive"


def _explanation(
    feature: str, verdict: Verdict, perm_p: float, n_perm: int, n_merged: int
) -> str:
    if verdict == "signal":
        text = (
            f"Knowing {feature} genuinely improves prediction: only {perm_p:.1%} of "
            f"{n_perm} random shuffles do as well, and the improvement stays above "
            f"the negligibility line."
        )
    elif verdict == "noise":
        text = (
            f"The association between {feature} and satisfaction is indistinguishable "
            f"from chance: {perm_p:.0%} of random shuffles produce an improvement at "
            f"least as large."
        )
    else:
        text = (
            f"The evidence for {feature} is not sufficient to call it either way: "
            f"neither a real effect nor pure chance can be ruled out."
        )
    if perm_p <= 1 / (n_perm + 1):
        text += f" (The p-value sits at the resolution floor of 1/{n_perm + 1}.)"
    if n_merged:
        noun = "category was" if n_merged == 1 else "categories were"
        text += f" ({n_merged} rare {noun} merged into one bucket before testing.)"
    return text


@dataclass(frozen=True)
class _RawEffect:
    feature: str
    delta: float
    ci_low: float
    ci_high: float
    perm_p: float
    null_mean: float
    null_p95: float
    n: int
    n_merged: int
    constant: bool
    native: tuple[NativeTest, float, str]


def _null_deltas(
    encoded: _Encoded, y: IntArray, feature_index: int, n_perm: int, seed: int
) -> FloatArray:
    rng = np.random.default_rng([seed, 2, feature_index])
    null = np.empty(n_perm)
    for p in range(n_perm):
        y_perm = np.asarray(rng.permutation(y), dtype=np.int64)
        null[p] = _oof_row_deltas(encoded, y_perm, seed).mean()
    return np.asarray(null, dtype=np.float64)


def _screen_one(
    df: pd.DataFrame,
    feature: str,
    y_all: IntArray,
    feature_index: int,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> _RawEffect:
    encoded = _encode(df[feature])
    mask = encoded.mask
    n = int(mask.sum())
    if n < len(df):
        logger.warning(
            "%s: %d rows with missing values excluded from screening; n=%d",
            feature,
            len(df) - n,
            n,
        )
    series = df[feature][mask]
    y = y_all[mask]
    masked = _Encoded(
        kind=encoded.kind,
        values=encoded.values[mask] if encoded.values is not None else None,
        codes=encoded.codes[mask] if encoded.codes is not None else None,
        n_levels=encoded.n_levels,
        n_merged=encoded.n_merged,
        mask=np.ones(n, dtype=bool),
    )

    if series.nunique(dropna=True) <= 1:
        return _RawEffect(
            feature=feature,
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            perm_p=1.0,
            null_mean=0.0,
            null_p95=0.0,
            n=n,
            n_merged=0,
            constant=True,
            native=("chi2", 0.0, "cramers_v"),
        )

    row_deltas = _oof_row_deltas(masked, y, seed)
    delta = float(row_deltas.mean())

    rng_boot = np.random.default_rng([seed, 1, feature_index])
    idx = rng_boot.integers(0, n, size=(n_boot, n))
    boot = row_deltas[idx].mean(axis=1)
    ci_low, ci_high = (float(q) for q in np.percentile(boot, [2.5, 97.5]))

    null = _null_deltas(masked, y, feature_index, n_perm, seed)
    perm_p = float((1 + (null >= delta).sum()) / (1 + n_perm))

    return _RawEffect(
        feature=feature,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        perm_p=perm_p,
        null_mean=float(null.mean()),
        null_p95=float(np.percentile(null, 95)),
        n=n,
        n_merged=encoded.n_merged,
        constant=False,
        native=_native_effect(series, y),
    )


def signal_report(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> SignalReport:
    """Screen every feature with the common statistic and issue verdicts.

    ``df`` is the training split; the held-out test set never enters
    univariate screening.
    """
    names = _feature_list(df, target, features)
    y_all = _target_codes(df, target)
    raw = [
        _screen_one(df, name, y_all, i, n_boot=n_boot, n_perm=n_perm, seed=seed)
        for i, name in enumerate(names)
    ]

    _, adjusted, _, _ = multipletests([r.perm_p for r in raw], method="fdr_bh")
    effects: list[EffectResult] = []
    for r, p_adj in zip(raw, adjusted, strict=True):
        if r.constant:
            verdict: Verdict = "noise"
            explanation = (
                f"{r.feature} is constant in this sample, so it cannot carry "
                f"information about satisfaction; no test was run."
            )
        else:
            verdict = _verdict(float(p_adj), r.perm_p, r.ci_low, r.ci_high)
            explanation = _explanation(r.feature, verdict, r.perm_p, n_perm, r.n_merged)
        native_test, native_size, native_name = r.native
        effects.append(
            EffectResult(
                feature=r.feature,
                delta_logloss=r.delta,
                delta_ci_low=r.ci_low,
                delta_ci_high=r.ci_high,
                permutation_p=r.perm_p,
                permutation_p_adj=float(p_adj),
                null_mean=r.null_mean,
                null_p95=r.null_p95,
                verdict=verdict,
                explanation=explanation,
                n=r.n,
                native_test=native_test,
                native_effect_size=native_size,
                native_effect_name=native_name,
            )
        )
    effects.sort(key=lambda e: e.delta_logloss, reverse=True)
    n_signal = sum(1 for e in effects if e.verdict == "signal")
    return SignalReport(
        effects=effects,
        n_features_tested=len(effects),
        n_signal=n_signal,
        family_wise_note=(
            f"Permutation p-values are Benjamini-Hochberg adjusted across all "
            f"{len(effects)} features, tested as one family."
        ),
        dataset_verdict="signal_present" if n_signal > 0 else "no_detectable_signal",
    )


def effect_for(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_boot: int = N_BOOT,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> EffectResult:
    """One feature's result. BH needs the family, so the full family is computed."""
    report = signal_report(df, target, n_boot=n_boot, n_perm=n_perm, seed=seed)
    for effect in report.effects:
        if effect.feature == feature:
            return effect
    available = ", ".join(e.feature for e in report.effects)
    raise ValueError(f"Unknown feature {feature!r}. Available: {available}.")


def permutation_null(
    df: pd.DataFrame,
    feature: str,
    target: str,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> FloatArray:
    """The permutation null distribution of delta_logloss for one feature."""
    names = _feature_list(df, target, None)
    if feature not in names:
        raise ValueError(f"Unknown feature {feature!r}. Available: {', '.join(names)}.")
    feature_index = names.index(feature)
    y_all = _target_codes(df, target)
    encoded = _encode(df[feature])
    mask = encoded.mask
    masked = _Encoded(
        kind=encoded.kind,
        values=encoded.values[mask] if encoded.values is not None else None,
        codes=encoded.codes[mask] if encoded.codes is not None else None,
        n_levels=encoded.n_levels,
        n_merged=encoded.n_merged,
        mask=np.ones(int(mask.sum()), dtype=bool),
    )
    return _null_deltas(masked, y_all[mask], feature_index, n_perm, seed)
