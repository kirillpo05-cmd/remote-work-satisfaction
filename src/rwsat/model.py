"""M4 — model layer: training, selection, calibration, prediction (SPEC M4).

Three primary models (dummy prior, ordinal logit, gradient boosting) plus a
diagnostic multinomial model fitted only when the proportional-odds assumption
is violated or statsmodels fails to converge. Log loss is the primary metric;
"beats baseline" and the selection tie rule both use the paired per-fold
standard deviation (D-009). The held-out test set is measured exactly once,
after selection has been made on cross-validation.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools import add_constant

from rwsat.config import (
    ALPHA,
    BASELINE_SD_MULTIPLIER,
    CONFIDENCE_DEVIATION_THRESHOLD,
    N_IMPORTANCE_REPEATS,
    N_PERM_MODEL,
    N_SPLITS,
    SEED,
)
from rwsat.data import SCHEMA, TARGET, split
from rwsat.features import build_preprocessor

logger = logging.getLogger(__name__)

_EPS = 1e-15
_COMPLEXITY = ["dummy", "ordinal_logit", "multinomial", "gradient_boosting"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class ModelCard:
    model_name: str
    trained_at: datetime
    n_train: int
    n_test: int
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    lift_over_baseline: dict[str, float]
    cv_mean: float
    cv_std: float
    permutation_test_p: float
    is_better_than_baseline: bool
    cv_fold_logloss: list[float]
    calibration: dict[str, list[tuple[float, float]]]
    proportional_odds_ok: bool | None
    feature_effects: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    probabilities: dict[str, float]
    predicted_class: str
    prior_probabilities: dict[str, float]
    max_deviation_from_prior: float
    confidence: str  # "high" | "low" | "not_better_than_guessing"
    warning: str | None
    flags: list[str]


@dataclass
class TrainedModel:
    name: str
    pipeline: Any  # fitted ColumnTransformer
    estimator: Any  # fitted estimator exposing predict_proba
    classes: list[str]
    prior: dict[str, float]
    fill_values: dict[str, Any]
    numeric_ranges: dict[str, tuple[float, float]]
    is_better_than_baseline: bool
    card: ModelCard


@dataclass
class TrainResult:
    cards: dict[str, ModelCard]
    selected: str
    model: TrainedModel


class _ModelUnavailable(Exception):
    """statsmodels failed hard; the model is excluded (SPEC M4 edge case)."""


class _OrderedLogit:
    """statsmodels OrderedModel behind the fit/predict_proba interface."""

    def __init__(self) -> None:
        self.converged = False
        self.llf = float("nan")
        self.n_params = 0
        self._result: Any = None

    def fit(self, x: FloatArray, y: IntArray) -> "_OrderedLogit":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._result = OrderedModel(y, x, distr="logit").fit(
                    method="bfgs", maxiter=500, disp=False
                )
        except Exception as exc:  # LinAlgError, ValueError from statsmodels
            raise _ModelUnavailable(f"ordinal logit failed to fit: {exc}") from exc
        self.converged = bool(self._result.mle_retvals.get("converged", False))
        self.llf = float(self._result.llf)
        self.n_params = int(self._result.params.size)
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return np.asarray(self._result.predict(x), dtype=np.float64)


def _make_estimator(name: str, seed: int) -> Any:
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name == "ordinal_logit":
        return _OrderedLogit()
    if name == "multinomial":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown model name {name!r}. Known: {', '.join(_COMPLEXITY)}.")


def _log_loss(y: IntArray, proba: FloatArray) -> float:
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], _EPS, None)).mean())


def _metric_block(y: IntArray, proba: FloatArray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    onehot = np.eye(proba.shape[1])[y]
    return {
        "log_loss": _log_loss(y, proba),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }


def _folds(y: IntArray, seed: int) -> list[tuple[IntArray, IntArray]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y)), y)]


def _cv(
    x: pd.DataFrame, y: IntArray, name: str, seed: int
) -> tuple[list[dict[str, float]], list[str]]:
    """Per-fold CV metrics. Preprocessor and estimator are refitted inside
    every fold; nothing is fitted outside a training fold."""
    per_fold: list[dict[str, float]] = []
    notes: list[str] = []
    for k, (tr, va) in enumerate(_folds(y, seed)):
        preprocessor = build_preprocessor(SCHEMA)
        x_tr = preprocessor.fit_transform(x.iloc[tr])
        x_va = preprocessor.transform(x.iloc[va])
        estimator = _make_estimator(name, seed)
        estimator.fit(x_tr, y[tr])
        if isinstance(estimator, _OrderedLogit) and not estimator.converged:
            notes.append(f"fold {k}: ordinal logit did not converge")
        per_fold.append(_metric_block(y[va], estimator.predict_proba(x_va)))
    return per_fold, notes


def _paired_indistinguishable(candidate: list[float], best: list[float]) -> bool:
    diff = np.asarray(candidate) - np.asarray(best)
    if np.allclose(diff, 0.0):
        return True
    return bool(diff.mean() <= BASELINE_SD_MULTIPLIER * diff.std(ddof=1))


def _paired_beats(baseline: list[float], candidate: list[float]) -> bool:
    diff = np.asarray(baseline) - np.asarray(candidate)
    if np.allclose(diff, 0.0):
        return False
    return bool(diff.mean() > BASELINE_SD_MULTIPLIER * diff.std(ddof=1))


def select_model(cards: dict[str, ModelCard]) -> str:
    """The simplest model whose CV log loss is statistically indistinguishable
    from the best one, judged on paired per-fold differences (D-009)."""
    present = [name for name in _COMPLEXITY if name in cards]
    if not present:
        raise ValueError("select_model needs at least one card.")
    best = min(present, key=lambda name: float(np.mean(cards[name].cv_fold_logloss)))
    for name in present:  # simplest first
        if _paired_indistinguishable(cards[name].cv_fold_logloss, cards[best].cv_fold_logloss):
            return name
    return best


def _proportional_odds_test(
    x_train_t: FloatArray, y_train: IntArray
) -> tuple[bool | None, _OrderedLogit | None, list[str]]:
    """Likelihood-ratio comparison of the ordinal fit against the multinomial
    fit (D-005). Returns (ok, fitted ordinal on full train, notes)."""
    notes: list[str] = []
    try:
        ordered = _OrderedLogit().fit(x_train_t, y_train)
    except _ModelUnavailable as exc:
        logger.warning("%s", exc)
        return None, None, [str(exc)]
    if not ordered.converged:
        notes.append("ordinal logit did not converge on the full training split")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mn = sm.MNLogit(y_train, add_constant(x_train_t)).fit(
                method="bfgs", maxiter=500, disp=False
            )
    except Exception as exc:
        notes.append(f"multinomial fit for the LR test failed: {exc}")
        return None, ordered, notes
    lr_stat = 2.0 * (float(mn.llf) - ordered.llf)
    df_diff = int(np.asarray(mn.params).size) - ordered.n_params
    p_value = float(chi2_dist.sf(max(lr_stat, 0.0), df_diff))
    return bool(p_value >= ALPHA), ordered, notes


def _calibration(
    y: IntArray, proba: FloatArray, classes: list[str]
) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    for j, cls in enumerate(classes):
        frac_pos, mean_pred = calibration_curve(
            (y == j).astype(int), proba[:, j], n_bins=10, strategy="uniform"
        )
        curves[cls] = [(float(p), float(t)) for p, t in zip(mean_pred, frac_pos, strict=True)]
    return curves


def _permutation_importance(
    preprocessor: Any,
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: IntArray,
    seed: int,
) -> list[dict[str, Any]]:
    """Permutation importance with percentile CIs over N_IMPORTANCE_REPEATS
    column shuffles. Importance = log-loss increase when the column is broken."""
    base = _log_loss(y_test, estimator.predict_proba(preprocessor.transform(x_test)))
    rng = np.random.default_rng([seed, 5])
    out: list[dict[str, Any]] = []
    for feature in [n for n, s in SCHEMA.items() if s.role == "feature"]:
        deltas = np.empty(N_IMPORTANCE_REPEATS)
        for r in range(N_IMPORTANCE_REPEATS):
            shuffled = x_test.copy()
            state = int(rng.integers(0, 2**31))
            shuffled[feature] = (
                x_test[feature].sample(frac=1.0, random_state=state).set_axis(x_test.index)
            )
            proba = estimator.predict_proba(preprocessor.transform(shuffled))
            deltas[r] = _log_loss(y_test, proba) - base
        out.append(
            {
                "feature": feature,
                "importance_mean": float(deltas.mean()),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
            }
        )
    return out


def _model_permutation_test(
    x_train: pd.DataFrame,
    y_train: IntArray,
    observed_cv_logloss: float,
    n_perm_model: int,
    seed: int,
) -> float:
    """Null CV log loss of the gradient boosting model under target shuffles
    (D-007). Add-one p-value (D-008)."""
    rng = np.random.default_rng([seed, 4])
    null = np.empty(n_perm_model)
    for p in range(n_perm_model):
        y_perm = np.asarray(rng.permutation(y_train), dtype=np.int64)
        losses = []
        for tr, va in _folds(y_perm, seed):
            preprocessor = build_preprocessor(SCHEMA)
            x_tr = preprocessor.fit_transform(x_train.iloc[tr])
            x_va = preprocessor.transform(x_train.iloc[va])
            clf = HistGradientBoostingClassifier(random_state=seed)
            clf.fit(x_tr, y_perm[tr])
            losses.append(_log_loss(y_perm[va], clf.predict_proba(x_va)))
        null[p] = float(np.mean(losses))
    return float((1 + (null <= observed_cv_logloss).sum()) / (1 + n_perm_model))


def train_all(
    df: pd.DataFrame, n_perm_model: int = N_PERM_MODEL, seed: int = SEED
) -> TrainResult:
    """Train the primary models (plus the diagnostic multinomial when its
    conditions hold), select on CV, then measure the test split exactly once."""
    train_df, test_df = split(df, seed=seed)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    classes = [str(c) for c in df[TARGET].cat.categories]
    y_train = np.asarray(train_df[TARGET].cat.codes, dtype=np.int64)
    y_test = np.asarray(test_df[TARGET].cat.codes, dtype=np.int64)
    x_train = train_df.drop(columns=[TARGET])
    x_test = test_df.drop(columns=[TARGET])

    # --- cross-validation on the training split only ---
    fold_metrics: dict[str, list[dict[str, float]]] = {}
    cv_notes: dict[str, list[str]] = {}
    ordinal_available = True
    for name in ("dummy", "ordinal_logit", "gradient_boosting"):
        try:
            fold_metrics[name], cv_notes[name] = _cv(x_train, y_train, name, seed)
        except _ModelUnavailable as exc:
            logger.warning("%s marked unavailable: %s", name, exc)
            ordinal_available = False
            cv_notes[name] = [str(exc)]

    # --- proportional-odds check on the full training split ---
    po_preprocessor = build_preprocessor(SCHEMA)
    x_train_t = np.asarray(po_preprocessor.fit_transform(x_train), dtype=np.float64)
    po_ok, _, po_notes = _proportional_odds_test(x_train_t, y_train)

    # The diagnostic multinomial model, only under its two conditions.
    if po_ok is False or not ordinal_available:
        fold_metrics["multinomial"], cv_notes["multinomial"] = _cv(
            x_train, y_train, "multinomial", seed
        )

    candidates = [n for n in _COMPLEXITY if n in fold_metrics]
    fold_logloss = {n: [f["log_loss"] for f in fold_metrics[n]] for n in candidates}

    # --- model-level permutation test (diagnostic, HGB, D-007) ---
    perm_p = _model_permutation_test(
        x_train, y_train, float(np.mean(fold_logloss["gradient_boosting"])), n_perm_model, seed
    )

    # --- final fits on the full training split, single test measurement ---
    trained: dict[str, tuple[Any, Any]] = {}
    cards: dict[str, ModelCard] = {}
    baseline_test: dict[str, float] = {}
    now = datetime.now(UTC)
    for name in candidates:
        preprocessor = build_preprocessor(SCHEMA)
        x_tr_t = preprocessor.fit_transform(x_train)
        estimator = _make_estimator(name, seed)
        estimator.fit(x_tr_t, y_train)
        trained[name] = (preprocessor, estimator)
        proba_test = np.asarray(
            estimator.predict_proba(preprocessor.transform(x_test)), dtype=np.float64
        )
        metrics = _metric_block(y_test, proba_test)
        if name == "dummy":
            baseline_test = metrics
        beats = _paired_beats(fold_logloss["dummy"], fold_logloss[name])
        card_warnings = list(cv_notes.get(name, []))
        if name == "ordinal_logit":
            card_warnings += po_notes
        if name != "dummy" and not beats:
            card_warnings.append(
                "This model does not beat the prior baseline; its predictions "
                "must not drive decisions about people."
            )
        card_warnings.append(
            "permutation_test_p is computed on gradient_boosting for every card (D-007)."
        )
        cards[name] = ModelCard(
            model_name=name,
            trained_at=now,
            n_train=len(train_df),
            n_test=len(test_df),
            metrics=metrics,
            baseline_metrics={},  # filled below, once the dummy card exists
            lift_over_baseline={},
            cv_mean=float(np.mean(fold_logloss[name])),
            cv_std=float(np.std(fold_logloss[name])),
            permutation_test_p=perm_p,
            is_better_than_baseline=beats,
            cv_fold_logloss=[float(v) for v in fold_logloss[name]],
            calibration=_calibration(y_test, proba_test, classes),
            proportional_odds_ok=po_ok if name == "ordinal_logit" else None,
            feature_effects=_permutation_importance(
                preprocessor, estimator, x_test, y_test, seed
            ),
            warnings=card_warnings,
        )
    for card in cards.values():
        card.baseline_metrics = dict(baseline_test)
        card.lift_over_baseline = {
            metric: (
                baseline_test[metric] - card.metrics[metric]
                if metric in ("log_loss", "brier")  # lower is better: lift = reduction
                else card.metrics[metric] - baseline_test[metric]
            )
            for metric in card.metrics
        }

    selected = select_model(cards)
    preprocessor, estimator = trained[selected]
    prior_freqs = np.bincount(y_train, minlength=len(classes)) / len(y_train)
    numeric_features = [n for n, s in SCHEMA.items() if s.role == "feature" and s.dtype == "int"]
    fill_values: dict[str, Any] = {}
    for name, spec in SCHEMA.items():
        if spec.role != "feature":
            continue
        if spec.dtype == "int":
            fill_values[name] = float(x_train[name].median())
        else:
            fill_values[name] = x_train[name].mode().iloc[0]
    model = TrainedModel(
        name=selected,
        pipeline=preprocessor,
        estimator=estimator,
        classes=classes,
        prior={cls: float(p) for cls, p in zip(classes, prior_freqs, strict=True)},
        fill_values=fill_values,
        numeric_ranges={
            n: (float(x_train[n].min()), float(x_train[n].max())) for n in numeric_features
        },
        is_better_than_baseline=cards[selected].is_better_than_baseline,
        card=cards[selected],
    )
    return TrainResult(cards=cards, selected=selected, model=model)


def save(model: TrainedModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> TrainedModel:
    if not path.exists():
        raise FileNotFoundError(
            f"No model artefact at {path}. Run: uv run python -m rwsat.cli train"
        )
    model = joblib.load(path)
    if not isinstance(model, TrainedModel):
        raise ValueError(f"{path} does not contain a TrainedModel.")
    return model


def predict_one(model: TrainedModel, payload: dict[str, Any]) -> Prediction:
    feature_specs = {n: s for n, s in SCHEMA.items() if s.role == "feature"}
    unknown_keys = sorted(set(payload) - set(feature_specs))
    if unknown_keys:
        raise ValueError(
            f"Unknown fields: {', '.join(unknown_keys)}. "
            f"Valid fields: {', '.join(feature_specs)}."
        )

    flags: list[str] = []
    row: dict[str, Any] = {}
    for name, spec in feature_specs.items():
        value = payload.get(name)
        if value is None:
            row[name] = model.fill_values[name]
            flags.append(f"imputed:{name}")
            continue
        if spec.dtype == "int":
            value = float(value)
            low, high = model.numeric_ranges[name]
            if not low <= value <= high and "out_of_range" not in flags:
                flags.append("out_of_range")
        else:
            levels = spec.order if spec.order is not None else spec.categories
            assert levels is not None
            if value not in levels and "unknown_category" not in flags:
                flags.append("unknown_category")
        row[name] = value

    frame = pd.DataFrame([row])
    for name, spec in feature_specs.items():
        if spec.dtype == "int":
            frame[name] = frame[name].astype(float)
        elif spec.order is not None:
            frame[name] = frame[name].astype(pd.CategoricalDtype(spec.order, ordered=True))
        else:
            frame[name] = frame[name].astype(pd.CategoricalDtype(spec.categories))

    proba = np.asarray(
        model.estimator.predict_proba(model.pipeline.transform(frame)), dtype=np.float64
    )[0]
    probabilities = {cls: float(p) for cls, p in zip(model.classes, proba, strict=True)}
    predicted = model.classes[int(np.argmax(proba))]
    max_deviation = float(max(abs(probabilities[c] - model.prior[c]) for c in model.classes))

    if not model.is_better_than_baseline:
        confidence = "not_better_than_guessing"
        warning: str | None = (
            "The selected model does not beat the prior baseline on this data. "
            "This prediction is essentially the class distribution and must not "
            "drive decisions about people."
        )
    else:
        confidence = "high" if max_deviation > CONFIDENCE_DEVIATION_THRESHOLD else "low"
        warning = None
        if "out_of_range" in flags and confidence == "high":
            confidence = "low"  # SPEC M2 edge case: out-of-range lowers confidence

    return Prediction(
        probabilities=probabilities,
        predicted_class=predicted,
        prior_probabilities=dict(model.prior),
        max_deviation_from_prior=max_deviation,
        confidence=confidence,
        warning=warning,
        flags=flags,
    )
