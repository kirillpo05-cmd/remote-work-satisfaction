"""Tests for M2 (SPEC M2). Written before the implementation.

The leakage test is the load-bearing one: it must actually catch a
preprocessor fitted on the full sample instead of the training split.
"""

import numpy as np
import pandas as pd

from rwsat.data import (
    SCHEMA,
    prepare,  # noqa: E402  (grouped for readability)
)
from rwsat.features import FeatureMeta, build_preprocessor, feature_metadata
from test_data import synthetic_frame


def _prepared(n: int = 600) -> pd.DataFrame:
    return prepare(synthetic_frame(n))


# --- leakage (mandatory) ---------------------------------------------------


def test_scaler_statistics_come_from_train_only() -> None:
    df = _prepared(600)
    # Make the train half and the full sample differ hard on Age.
    df["Age"] = np.concatenate([np.full(300, 25), np.full(300, 55)]).astype("int64")
    train = df.iloc[:300]

    preprocessor = build_preprocessor(SCHEMA)
    preprocessor.fit(train.drop(columns=["Satisfaction_with_Remote_Work"]))

    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    numeric_columns = preprocessor.transformers_[0][2]
    age_position = list(numeric_columns).index("Age")

    train_mean = float(train["Age"].mean())  # 25.0
    full_mean = float(df["Age"].mean())  # 40.0
    fitted_mean = float(scaler.mean_[age_position])
    assert fitted_mean == train_mean
    assert fitted_mean != full_mean


# --- metadata ----------------------------------------------------------------


def test_feature_metadata_covers_all_features_with_correct_kinds() -> None:
    metas = feature_metadata(SCHEMA)
    by_name: dict[str, FeatureMeta] = {m.name: m for m in metas}
    assert len(metas) == 18

    assert by_name["Age"].kind == "numeric"
    assert by_name["Age"].min == 22 and by_name["Age"].max == 60
    assert by_name["Age"].default is not None
    assert by_name["Age"].options is None

    assert by_name["Stress_Level"].kind == "ordinal"
    assert by_name["Stress_Level"].options == ["Low", "Medium", "High"]

    assert by_name["Gender"].kind == "categorical"
    assert by_name["Gender"].options == ["Female", "Male", "Non-binary", "Prefer not to say"]

    assert by_name["Access_to_Mental_Health_Resources"].kind == "categorical"
    assert all(m.label and "_" not in m.label for m in metas)


# --- encoding behaviour --------------------------------------------------------


def test_unknown_category_at_inference_does_not_crash() -> None:
    df = _prepared(300)
    x = df.drop(columns=["Satisfaction_with_Remote_Work"])
    preprocessor = build_preprocessor(SCHEMA)
    preprocessor.fit(x)

    probe = x.iloc[[0]].copy()
    # An unknown level arrives as NaN after the categorical cast; the pipeline
    # must transform it without raising (M2 edge case).
    probe["Gender"] = pd.Series(["Alien"], index=probe.index).astype(
        pd.CategoricalDtype(SCHEMA["Gender"].categories)
    )
    transformed = preprocessor.transform(probe)
    assert transformed.shape[0] == 1
    assert np.isfinite(np.asarray(transformed, dtype=float)).all()


def test_ordinal_encoding_respects_declared_order() -> None:
    df = _prepared(300)
    x = df.drop(columns=["Satisfaction_with_Remote_Work"])
    preprocessor = build_preprocessor(SCHEMA)
    preprocessor.fit(x)

    low = x.iloc[[0]].copy()
    high = x.iloc[[0]].copy()
    low["Stress_Level"] = pd.Categorical(
        ["Low"], categories=["Low", "Medium", "High"], ordered=True
    )
    high["Stress_Level"] = pd.Categorical(
        ["High"], categories=["Low", "Medium", "High"], ordered=True
    )
    t_low = np.asarray(preprocessor.transform(low), dtype=float).ravel()
    t_high = np.asarray(preprocessor.transform(high), dtype=float).ravel()

    diff = t_high - t_low
    changed = np.flatnonzero(diff != 0)
    assert len(changed) == 1  # only the ordinal slot moved
    assert diff[changed[0]] == 2.0  # High(2) - Low(0), hand-declared order
