"""M2 — feature layer: one preprocessor, identical at training and inference (SPEC M2).

Every transformer lives inside the pipeline handed to cross-validation; nothing
is fitted outside a training fold. The imputers exist for inference payloads
with missing fields — the raw CSV has no missing values once M1 loads it.
"""

from dataclasses import dataclass
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from rwsat.data import ColumnSpec


@dataclass(frozen=True)
class FeatureMeta:
    name: str
    kind: Literal["numeric", "categorical", "ordinal"]
    options: list[str] | None
    min: int | None
    max: int | None
    default: int | None
    label: str


def _kind(spec: ColumnSpec) -> Literal["numeric", "categorical", "ordinal"]:
    if spec.dtype in ("int", "float"):
        return "numeric"
    if spec.dtype == "ord":
        return "ordinal"
    return "categorical"  # cat and bin


def build_preprocessor(schema: dict[str, ColumnSpec]) -> ColumnTransformer:
    features = [spec for spec in schema.values() if spec.role == "feature"]
    numeric = [spec.name for spec in features if _kind(spec) == "numeric"]
    ordinal = [spec for spec in features if _kind(spec) == "ordinal"]
    categorical = [spec for spec in features if _kind(spec) == "categorical"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # Order is declared by hand in SCHEMA, never inferred from the data.
    # Unknown or missing levels encode to -1: the prediction still goes
    # through, and predict_one attaches the corresponding flag.
    ordinal_encoder = OrdinalEncoder(
        categories=[spec.order for spec in ordinal],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[spec.categories for spec in categorical],
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,  # statsmodels needs dense; the data is small
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("ordinal", ordinal_encoder, [spec.name for spec in ordinal]),
            ("categorical", categorical_pipeline, [spec.name for spec in categorical]),
        ]
    )


def feature_metadata(schema: dict[str, ColumnSpec]) -> list[FeatureMeta]:
    metas: list[FeatureMeta] = []
    for spec in schema.values():
        if spec.role != "feature":
            continue
        kind = _kind(spec)
        options: list[str] | None = None
        if kind != "numeric":
            levels = spec.order if spec.order is not None else spec.categories
            assert levels is not None, f"{spec.name}: non-numeric feature without levels"
            options = [str(level) for level in levels]
        default: int | None = None
        if kind == "numeric" and spec.minimum is not None and spec.maximum is not None:
            default = (spec.minimum + spec.maximum) // 2
        metas.append(
            FeatureMeta(
                name=spec.name,
                kind=kind,
                options=options,
                min=spec.minimum,
                max=spec.maximum,
                default=default,
                label=spec.name.replace("_", " "),
            )
        )
    return metas
