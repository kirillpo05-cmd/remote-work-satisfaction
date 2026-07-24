"""Edge-case tests for M1 (SPEC M1). Written before the implementation.

Synthetic fixtures cover the duplicate-ID and null cases because the real
file has neither.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from rwsat import data
from rwsat.config import RAW_PATH, SEED, TEST_SIZE
from rwsat.data import SCHEMA, TARGET, load_raw, prepare, split, validate


def synthetic_frame(n: int = 60) -> pd.DataFrame:
    """A small frame that conforms to SCHEMA, cycling through valid values."""
    rows = {}
    for name, spec in SCHEMA.items():
        if spec.name == "Employee_ID":
            rows[name] = [f"EMP{i:04d}" for i in range(n)]
        elif spec.dtype == "int":
            rows[name] = [20 + (i % 10) for i in range(n)]
        else:
            levels = spec.order if spec.order is not None else spec.categories
            assert levels is not None
            rows[name] = [levels[i % len(levels)] for i in range(n)]
    return pd.DataFrame(rows)


# --- load_raw: raises on unusable input ---------------------------------


def test_load_raw_missing_file_names_path_and_download_step(tmp_path: Path) -> None:
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError) as exc:
        load_raw(missing)
    message = str(exc.value)
    assert str(missing) in message
    assert "kaggle" in message.lower() or "download" in message.lower()


def test_load_raw_empty_file_raises_value_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_raw(empty)


def test_load_raw_header_only_raises_value_error(tmp_path: Path) -> None:
    header_only = tmp_path / "header.csv"
    header_only.write_text(",".join(SCHEMA) + "\n")
    with pytest.raises(ValueError):
        load_raw(header_only)


def test_load_raw_real_file_none_strings_survive() -> None:
    df = load_raw(RAW_PATH)
    assert (df["Mental_Health_Condition"] == "None").sum() == 1196
    assert (df["Physical_Activity"] == "None").sum() == 1629
    assert int(df.isna().sum().sum()) == 0


def test_load_raw_asserts_when_none_is_lost(tmp_path: Path) -> None:
    # A file whose "None" values would vanish under default na parsing.
    df = synthetic_frame()
    df["Mental_Health_Condition"] = "Anxiety"  # no literal "None" anywhere
    path = tmp_path / "no_none.csv"
    df.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_raw(path)


# --- validate: returns a report, never raises ----------------------------


def test_validate_missing_column_invalidates() -> None:
    df = synthetic_frame().drop(columns=["Stress_Level"])
    report = validate(df)
    assert "Stress_Level" in report.missing_columns
    assert report.is_valid is False


def test_validate_unexpected_category_listed_and_preserved() -> None:
    df = synthetic_frame()
    df.loc[0, "Physical_Activity"] = "Sometimes"
    report = validate(df)
    assert "Sometimes" in report.unexpected_categories.get("Physical_Activity", [])
    assert (df["Physical_Activity"] == "Sometimes").sum() == 1  # not mutated


def test_validate_duplicate_employee_id_warns_but_keeps_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = synthetic_frame()
    df.loc[1, "Employee_ID"] = df.loc[0, "Employee_ID"]
    with caplog.at_level(logging.WARNING, logger=data.__name__):
        validate(df)
    assert any("Employee_ID" in r.message for r in caplog.records)
    assert len(df) == 60  # rows are never dropped by validate


def test_validate_counts_nulls() -> None:
    df = synthetic_frame()
    df.loc[0, "Sleep_Quality"] = None
    report = validate(df)
    assert report.null_counts.get("Sleep_Quality") == 1


def test_validate_never_raises_on_garbage() -> None:
    garbage = pd.DataFrame({"foo": [1, 2], "bar": ["x", "y"]})
    report = validate(garbage)
    assert report.is_valid is False
    assert "foo" in report.unexpected_columns


def test_validate_dtype_mismatch_invalidates() -> None:
    df = synthetic_frame()
    df["Age"] = df["Age"].astype(str)
    report = validate(df)
    assert "Age" in report.dtype_mismatches
    assert report.is_valid is False


# --- prepare -------------------------------------------------------------


def test_prepare_drops_identifier_and_orders_categories() -> None:
    out = prepare(synthetic_frame())
    assert "Employee_ID" not in out.columns
    stress = out["Stress_Level"]
    assert isinstance(stress.dtype, pd.CategoricalDtype)
    assert stress.dtype.ordered
    assert list(stress.dtype.categories) == ["Low", "Medium", "High"]
    target = out[TARGET]
    assert isinstance(target.dtype, pd.CategoricalDtype)
    assert target.dtype.ordered
    assert list(target.dtype.categories) == ["Unsatisfied", "Neutral", "Satisfied"]


# --- split ---------------------------------------------------------------


def test_split_is_stratified_seeded_and_sized() -> None:
    df = prepare(load_raw(RAW_PATH))
    train_a, test_a = split(df, test_size=TEST_SIZE, seed=SEED)
    train_b, test_b = split(df, test_size=TEST_SIZE, seed=SEED)
    assert len(test_a) == round(len(df) * TEST_SIZE)
    assert len(train_a) + len(test_a) == len(df)
    # identical under the same seed
    assert train_a.index.equals(train_b.index)
    assert test_a.index.equals(test_b.index)
    # stratified: class shares within 1 percentage point of the full frame
    full = df[TARGET].value_counts(normalize=True)
    test_shares = test_a[TARGET].value_counts(normalize=True)
    assert (full - test_shares).abs().max() < 0.01
