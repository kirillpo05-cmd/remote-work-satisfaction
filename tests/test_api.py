"""TestClient coverage for M5 (SPEC M5), every error branch included.

The app is pointed at self-contained artifacts built from synthetic data —
no test depends on the real trained model or the real signal report.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rwsat import config
from rwsat.cli import _card_as_dict
from rwsat.config import SEED
from rwsat.data import TARGET as REAL_TARGET
from rwsat.model import save, train_all
from rwsat.stats import signal_report
from test_model import random_frame
from test_stats import TARGET as SYN_TARGET
from test_stats import u_and_noise_frame


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny but complete artifacts: model + cards from a random-target frame,
    signal report from the U-shape/noise frame."""
    directory = tmp_path_factory.mktemp("artifacts")
    result = train_all(random_frame(), n_perm_model=2, seed=SEED)
    save(result.model, directory / "model.joblib")
    (directory / "model_cards.json").write_text(
        json.dumps(
            {
                "selected": result.selected,
                "cards": {n: _card_as_dict(c) for n, c in result.cards.items()},
            }
        )
    )
    report = signal_report(u_and_noise_frame(n=400, seed=3), SYN_TARGET, n_boot=50, n_perm=20)
    (directory / "signal_report.json").write_text(json.dumps(asdict(report)))
    return directory


def _client(monkeypatch: pytest.MonkeyPatch, directory: Path) -> TestClient:
    from rwsat.api import create_app

    monkeypatch.setattr(config, "MODEL_PATH", directory / "model.joblib")
    monkeypatch.setattr(config, "MODEL_CARDS_PATH", directory / "model_cards.json")
    monkeypatch.setattr(config, "SIGNAL_REPORT_PATH", directory / "signal_report.json")
    return TestClient(create_app())


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, artifact_dir: Path) -> TestClient:
    with _client(monkeypatch, artifact_dir) as c:
        yield c


@pytest.fixture()
def bare_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """An app started with no artifacts at all."""
    with _client(monkeypatch, tmp_path) as c:
        yield c


# --- health -----------------------------------------------------------------


def test_health_reflects_real_state(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "data_loaded": True}


def test_health_with_no_artifacts_still_serves(bare_client: TestClient) -> None:
    response = bare_client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


# --- schema -------------------------------------------------------------------


def test_schema_serves_the_form_contract(client: TestClient) -> None:
    body = client.get("/schema").json()
    assert len(body["features"]) == 18
    assert body["target"]["name"] == REAL_TARGET
    assert body["target"]["classes"] == ["Unsatisfied", "Neutral", "Satisfied"]
    by_name = {f["name"]: f for f in body["features"]}
    assert by_name["Age"]["kind"] == "numeric"
    assert by_name["Age"]["min"] == 22
    assert by_name["Stress_Level"]["options"] == ["Low", "Medium", "High"]


# --- effects --------------------------------------------------------------------


def test_effects_ranked_by_delta_descending(client: TestClient) -> None:
    body = client.get("/effects").json()
    deltas = [e["delta_logloss"] for e in body["effects"]]
    assert deltas == sorted(deltas, reverse=True)
    assert body["dataset_verdict"] in ("signal_present", "no_detectable_signal")
    assert body["effects"][0]["null_deltas"]  # histogram source ships with the report


def test_effect_by_name_and_404(client: TestClient) -> None:
    assert client.get("/effects/u_feature").status_code == 200
    response = client.get("/effects/Nonexistent")
    assert response.status_code == 404


def test_effects_503_names_the_fix(bare_client: TestClient) -> None:
    response = bare_client.get("/effects")
    assert response.status_code == 503
    assert "rwsat.cli verify-signal" in response.json()["detail"]


# --- predict ----------------------------------------------------------------------


def test_predict_fills_missing_and_flags(client: TestClient) -> None:
    response = client.post("/predict", json={"features": {"Age": 30}})
    assert response.status_code == 200
    body = response.json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert "imputed:Gender" in body["flags"]
    assert body["warning"]  # random-target model: warning is mandatory


def test_predict_422_lists_field_reasons(client: TestClient) -> None:
    response = client.post("/predict", json={"features": {"Age": "not a number"}})
    assert response.status_code == 422
    assert "Age" in json.dumps(response.json())

    response = client.post("/predict", json={"features": {"Nonsense_Field": 1}})
    assert response.status_code == 422
    assert "Nonsense_Field" in json.dumps(response.json())


def test_predict_503_names_the_fix(bare_client: TestClient) -> None:
    response = bare_client.post("/predict", json={"features": {}})
    assert response.status_code == 503
    assert "rwsat.cli train" in response.json()["detail"]


# --- model cards ----------------------------------------------------------------


def test_model_card_selected_and_all(client: TestClient) -> None:
    selected = client.get("/model-card").json()
    assert selected["model_name"] == "dummy"
    everyone = client.get("/model-card/all").json()
    assert {"dummy", "ordinal_logit", "gradient_boosting"} <= set(everyone)


def test_model_card_503_names_the_fix(bare_client: TestClient) -> None:
    response = bare_client.get("/model-card")
    assert response.status_code == 503
    assert "rwsat.cli train" in response.json()["detail"]


# --- validation --------------------------------------------------------------------


def test_validation_report_served(client: TestClient) -> None:
    body = client.get("/validation").json()
    assert body["is_valid"] is True
    assert body["missing_columns"] == []
