"""M5 — FastAPI service layer (SPEC M5).

The API serves, it does not compute: the signal report and the model are
precomputed artifacts loaded once via lifespan. A missing artifact never takes
the service down — the affected endpoint returns 503 with the exact command
that fixes it.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rwsat import config
from rwsat.data import SCHEMA, TARGET, ValidationReport, load_raw, validate
from rwsat.features import feature_metadata
from rwsat.model import TrainedModel, load, predict_one

logger = logging.getLogger(__name__)

TRAIN_FIX = "Run: uv run python -m rwsat.cli train"
SIGNAL_FIX = "Run: uv run python -m rwsat.cli verify-signal"
DATA_FIX = (
    "Download 'waqi786/remote-work-and-mental-health' from Kaggle and place the "
    "CSV at data/raw/Impact_of_Remote_Work_on_Mental_Health.csv"
)


# --- response contracts (every body is a pydantic model) --------------------


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool


class FeatureMetaOut(BaseModel):
    name: str
    kind: Literal["numeric", "categorical", "ordinal"]
    options: list[str] | None
    min: int | None
    max: int | None
    default: int | None
    label: str


class TargetOut(BaseModel):
    name: str
    classes: list[str]


class SchemaOut(BaseModel):
    features: list[FeatureMetaOut]
    target: TargetOut


class EffectOut(BaseModel):
    feature: str
    delta_logloss: float
    delta_ci_low: float
    delta_ci_high: float
    permutation_p: float
    permutation_p_adj: float
    null_mean: float
    null_p95: float
    null_deltas: list[float]
    verdict: Literal["signal", "inconclusive", "noise"]
    explanation: str
    n: int
    native_test: Literal["chi2", "kruskal", "spearman"]
    native_effect_size: float
    native_effect_name: str


class SignalReportOut(BaseModel):
    effects: list[EffectOut]
    n_features_tested: int
    n_signal: int
    family_wise_note: str
    dataset_verdict: Literal["signal_present", "no_detectable_signal"]


class PredictIn(BaseModel):
    features: dict[str, Any]


class PredictionOut(BaseModel):
    probabilities: dict[str, float]
    predicted_class: str
    prior_probabilities: dict[str, float]
    max_deviation_from_prior: float
    confidence: Literal["high", "low", "not_better_than_guessing"]
    warning: str | None
    flags: list[str]


class ModelCardOut(BaseModel):
    model_name: str
    trained_at: str
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
    warnings: list[str]


class ValidationOut(BaseModel):
    missing_columns: list[str]
    unexpected_columns: list[str]
    dtype_mismatches: dict[str, str]
    unexpected_categories: dict[str, list[str]]
    null_counts: dict[str, int]
    is_valid: bool


def _load_state(app: FastAPI) -> None:
    """Read every artifact once. Paths are read from config at call time."""
    validation_report: ValidationReport | None = None
    data_loaded = False
    try:
        raw = load_raw()
        validation_report = validate(raw)
        data_loaded = bool(validation_report.is_valid)
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        logger.warning("Raw data unavailable: %s", exc)
    app.state.validation = validation_report
    app.state.data_loaded = data_loaded

    signal: dict[str, Any] | None = None
    if config.SIGNAL_REPORT_PATH.exists():
        signal = json.loads(config.SIGNAL_REPORT_PATH.read_text())
    app.state.signal = signal

    model: TrainedModel | None = None
    if config.MODEL_PATH.exists():
        model = load(config.MODEL_PATH)
    app.state.model = model

    cards: dict[str, Any] | None = None
    if config.MODEL_CARDS_PATH.exists():
        cards = json.loads(config.MODEL_CARDS_PATH.read_text())
    app.state.cards = cards


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _load_state(app)
        yield

    app = FastAPI(title="RWSAT", lifespan=lifespan)

    @app.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        return HealthOut(
            status="ok",
            model_loaded=app.state.model is not None,
            data_loaded=app.state.data_loaded,
        )

    @app.get("/schema", response_model=SchemaOut)
    def schema() -> SchemaOut:
        target_spec = SCHEMA[TARGET]
        assert target_spec.order is not None
        return SchemaOut(
            features=[FeatureMetaOut(**asdict(meta)) for meta in feature_metadata(SCHEMA)],
            target=TargetOut(name=TARGET, classes=[str(c) for c in target_spec.order]),
        )

    @app.get("/effects", response_model=SignalReportOut)
    def effects() -> Any:
        if app.state.signal is None:
            raise HTTPException(503, detail=f"No signal report cached. {SIGNAL_FIX}")
        return app.state.signal

    @app.get("/effects/{name}", response_model=EffectOut)
    def effect_by_name(name: str) -> Any:
        if app.state.signal is None:
            raise HTTPException(503, detail=f"No signal report cached. {SIGNAL_FIX}")
        for effect in app.state.signal["effects"]:
            if effect["feature"] == name:
                return effect
        raise HTTPException(404, detail=f"No effect result for feature {name!r}.")

    @app.post("/predict", response_model=PredictionOut)
    def predict(body: PredictIn) -> PredictionOut:
        if app.state.model is None:
            raise HTTPException(503, detail=f"No trained model artefact. {TRAIN_FIX}")
        feature_specs = {n: s for n, s in SCHEMA.items() if s.role == "feature"}
        field_errors = [
            {"field": name, "reason": "unknown field"}
            for name in body.features
            if name not in feature_specs
        ]
        for name, value in body.features.items():
            spec = feature_specs.get(name)
            if spec is not None and spec.dtype in ("int", "float") and value is not None:
                try:
                    float(value)
                except (TypeError, ValueError):
                    field_errors.append(
                        {"field": name, "reason": f"expected a number, got {value!r}"}
                    )
        if field_errors:
            raise HTTPException(422, detail=field_errors)
        try:
            prediction = predict_one(app.state.model, body.features)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return PredictionOut(**asdict(prediction))

    @app.get("/model-card", response_model=ModelCardOut)
    def model_card() -> Any:
        if app.state.cards is None:
            raise HTTPException(503, detail=f"No model cards cached. {TRAIN_FIX}")
        return app.state.cards["cards"][app.state.cards["selected"]]

    @app.get("/model-card/all", response_model=dict[str, ModelCardOut])
    def model_card_all() -> Any:
        if app.state.cards is None:
            raise HTTPException(503, detail=f"No model cards cached. {TRAIN_FIX}")
        return app.state.cards["cards"]

    @app.get("/validation", response_model=ValidationOut)
    def validation() -> Any:
        if app.state.validation is None:
            raise HTTPException(503, detail=f"Raw data not found. {DATA_FIX}")
        return asdict(app.state.validation)

    return app


app = create_app()
