"""M1 — data layer: schema, loading, validation, preparation, split (SPEC M1).

Raise/report boundary: `load_raw` raises on unusable input (missing file,
empty frame); `validate` returns a report on schema problems and never raises.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from rwsat.config import RAW_PATH, SEED, TEST_SIZE

logger = logging.getLogger(__name__)

TARGET = "Satisfaction_with_Remote_Work"

# Columns whose literal string "None" is a meaningful category, not missing
# data. pandas' default na_values would silently turn them into NaN.
NONE_IS_A_CATEGORY = ("Mental_Health_Condition", "Physical_Activity")

RATING_ORDER = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: Literal["int", "float", "cat", "ord", "bin"]
    role: Literal["feature", "target", "identifier", "dropped"]
    categories: list[str] | None = None
    order: list[str] | list[int] | None = None


def _feature(
    name: str,
    dtype: Literal["int", "float", "cat", "ord", "bin"],
    categories: list[str] | None = None,
    order: list[str] | list[int] | None = None,
) -> ColumnSpec:
    return ColumnSpec(name=name, dtype=dtype, role="feature", categories=categories, order=order)


SCHEMA: dict[str, ColumnSpec] = {
    spec.name: spec
    for spec in [
        ColumnSpec(name="Employee_ID", dtype="cat", role="identifier"),
        _feature("Age", "int"),
        _feature("Gender", "cat", categories=["Female", "Male", "Non-binary", "Prefer not to say"]),
        _feature(
            "Job_Role",
            "cat",
            categories=[
                "Data Scientist",
                "Designer",
                "HR",
                "Marketing",
                "Project Manager",
                "Sales",
                "Software Engineer",
            ],
        ),
        _feature(
            "Industry",
            "cat",
            categories=[
                "Consulting",
                "Education",
                "Finance",
                "Healthcare",
                "IT",
                "Manufacturing",
                "Retail",
            ],
        ),
        _feature("Years_of_Experience", "int"),
        _feature("Work_Location", "cat", categories=["Hybrid", "Onsite", "Remote"]),
        _feature("Hours_Worked_Per_Week", "int"),
        _feature("Number_of_Virtual_Meetings", "int"),
        _feature("Work_Life_Balance_Rating", "ord", order=RATING_ORDER),
        _feature("Stress_Level", "ord", order=["Low", "Medium", "High"]),
        _feature(
            "Mental_Health_Condition",
            "cat",
            categories=["Anxiety", "Burnout", "Depression", "None"],
        ),
        _feature("Access_to_Mental_Health_Resources", "bin", categories=["No", "Yes"]),
        _feature("Productivity_Change", "ord", order=["Decrease", "No Change", "Increase"]),
        _feature("Social_Isolation_Rating", "ord", order=RATING_ORDER),
        ColumnSpec(
            name=TARGET,
            dtype="ord",
            role="target",
            order=["Unsatisfied", "Neutral", "Satisfied"],
        ),
        _feature("Company_Support_for_Remote_Work", "ord", order=RATING_ORDER),
        _feature("Physical_Activity", "ord", order=["None", "Weekly", "Daily"]),
        _feature("Sleep_Quality", "ord", order=["Poor", "Average", "Good"]),
        _feature(
            "Region",
            "cat",
            categories=["Africa", "Asia", "Europe", "North America", "Oceania", "South America"],
        ),
    ]
}


@dataclass
class ValidationReport:
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    dtype_mismatches: dict[str, str] = field(default_factory=dict)
    unexpected_categories: dict[str, list[str]] = field(default_factory=dict)
    null_counts: dict[str, int] = field(default_factory=dict)
    is_valid: bool = True


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises on unusable input; schema problems go to validate().

    keep_default_na=False is load-bearing: the literal string "None" is a
    meaningful category in two columns and must survive parsing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'waqi786/remote-work-and-mental-health' "
            f"from Kaggle and place the CSV at that path."
        )
    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path} is empty: no header, no rows.") from exc
    if df.empty:
        raise ValueError(f"{path} contains a header but no rows.")

    for column in NONE_IS_A_CATEGORY:
        if column not in df.columns:
            continue  # a missing column is validate()'s finding, not a load failure
        none_count = int((df[column] == "None").sum())
        nan_count = int(df[column].isna().sum())
        if none_count == 0 or nan_count > 0:
            raise AssertionError(
                f"The literal string 'None' did not survive loading {column} "
                f"(found {none_count} 'None', {nan_count} NaN). This means "
                f"keep_default_na=False was lost — pandas is converting a "
                f"meaningful category into missing data."
            )
    return df


def _expects_numeric(spec: ColumnSpec) -> bool:
    if spec.dtype in ("int", "float"):
        return True
    return spec.order is not None and all(isinstance(level, int) for level in spec.order)


def _declared_levels(spec: ColumnSpec) -> list[str] | list[int] | None:
    return spec.order if spec.order is not None else spec.categories


def validate(df: pd.DataFrame) -> ValidationReport:
    """Check df against SCHEMA. Returns a report; never raises."""
    report = ValidationReport()
    report.missing_columns = [name for name in SCHEMA if name not in df.columns]
    report.unexpected_columns = [name for name in df.columns if name not in SCHEMA]

    for name, spec in SCHEMA.items():
        if name in report.missing_columns:
            continue
        column = df[name]

        nulls = int(column.isna().sum())
        if nulls:
            report.null_counts[name] = nulls

        if _expects_numeric(spec):
            if not pd.api.types.is_numeric_dtype(column):
                report.dtype_mismatches[name] = f"expected numeric, found {column.dtype}"
        elif pd.api.types.is_numeric_dtype(column):
            report.dtype_mismatches[name] = f"expected string, found {column.dtype}"

        levels = _declared_levels(spec)
        if levels is not None and name not in report.dtype_mismatches:
            observed = set(column.dropna().unique())
            unexpected = sorted(str(value) for value in observed - set(levels))
            if unexpected:
                report.unexpected_categories[name] = unexpected
                logger.warning(
                    "Column %s contains categories not in SCHEMA: %s "
                    "(values preserved, not dropped)",
                    name,
                    unexpected,
                )

    if "Employee_ID" in df.columns:
        duplicates = int(df["Employee_ID"].duplicated().sum())
        if duplicates:
            logger.warning(
                "Employee_ID has %d duplicated values; rows are kept — dropping "
                "them requires an explicit decision.",
                duplicates,
            )

    report.is_valid = not report.missing_columns and not report.dtype_mismatches
    return report


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the identifier, cast dtypes, apply declared category orders.

    Assumes a frame that validate() would accept; unexpected values in ordered
    columns become NaN, which validate() reports beforehand.
    """
    out = df.copy()
    for name, spec in SCHEMA.items():
        if name not in out.columns:
            continue
        if spec.role == "identifier":
            out = out.drop(columns=[name])
        elif spec.dtype == "int":
            out[name] = out[name].astype("int64")
        elif spec.order is not None:
            out[name] = out[name].astype(pd.CategoricalDtype(spec.order, ordered=True))
        elif spec.categories is not None:
            out[name] = out[name].astype(pd.CategoricalDtype(spec.categories, ordered=False))
    return out


def split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the target. Happens before any preprocessing."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train_df, test_df
