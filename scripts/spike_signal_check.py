"""Phase 0 spike: the verify-signal protocol, run end to end.

Sanctioned under the CLAUDE.md Rule 1 exception. THROWAWAY CODE: this script
must be deleted or absorbed into spec'd modules before Phase 2 ends, and no
spec'd module may import from it. It imports rwsat.config and rwsat.data
(the dependency may only point in that direction).

Protocol (per .claude/skills/verify-signal and SPEC M3):
  1. Dummy baselines (prior, stratified), OOF metrics on the training split.
  2. Common statistic per feature: delta_logloss with bootstrap percentile CI,
     permutation null (refit per permutation), BH across the family, verdicts.
     Native effect sizes as descriptive columns.
  3. Ordinal logit and HistGradientBoosting under stratified 5-fold CV vs
     baseline; log loss decides.
  4. Model-level permutation test (diagnostic; never alters the verdict).
  5. dataset_verdict from the per-feature common statistic alone.

The held-out test set is never touched.
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

from rwsat.config import (
    ALPHA,
    ARTIFACTS_DIR,
    BASELINE_SD_MULTIPLIER,
    DELTA_NEGLIGIBLE,
    MIN_CATEGORY_N,
    N_BOOT,
    N_PERM,
    N_PERM_MODEL,
    N_SPLITS,
    N_UNIVARIATE_BINS,
    NOISE_PERMUTATION_P,
    SEED,
)
from rwsat.data import SCHEMA, TARGET, load_raw, prepare, split, validate

EPS = 1e-15
CLASSES = np.array([0, 1, 2])


def row_log_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(y)), y], EPS, None))


def metric_block(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = proba.argmax(axis=1)
    return {
        "log_loss": float(row_log_loss(proba, y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------- step 2 ---


def numeric_bin_codes(x_tr: np.ndarray, x_all: np.ndarray) -> np.ndarray:
    """Quantile-bin x_all using edges fitted on the fold-train values only."""
    quantiles = np.linspace(0, 1, N_UNIVARIATE_BINS + 1)[1:-1]
    edges = np.quantile(x_tr, quantiles)
    return np.searchsorted(edges, x_all, side="right")


def oof_row_deltas(
    values: np.ndarray | None, codes: np.ndarray | None, n_levels: int, y: np.ndarray
) -> np.ndarray:
    """Per-row (baseline loss - univariate model loss), out of fold.

    Numeric features pass `values` (binned per fold); others pass fixed `codes`.
    """
    n = len(y)
    base_ll = np.empty(n)
    model_ll = np.empty(n)
    for tr, va in folds(y):
        if values is not None:
            all_codes = numeric_bin_codes(values[tr], values)
            c_tr, c_va = all_codes[tr], all_codes[va]
            levels = N_UNIVARIATE_BINS
        else:
            assert codes is not None
            c_tr, c_va = codes[tr], codes[va]
            levels = n_levels
        eye = np.eye(levels)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(eye[c_tr], y[tr])
        proba = clf.predict_proba(eye[c_va])
        freqs = np.bincount(y[tr], minlength=3) / len(tr)
        model_ll[va] = row_log_loss(proba, y[va])
        base_ll[va] = -np.log(np.clip(freqs[y[va]], EPS, None))
    return base_ll - model_ll


def native_effect(name: str, series: pd.Series, y: np.ndarray) -> tuple[str, float, str]:
    spec = SCHEMA[name]
    n = len(y)
    if spec.dtype == "int":
        x = series.to_numpy(dtype=float)
        groups = [x[y == c] for c in CLASSES]
        h_stat, _ = sps.kruskal(*groups)
        eta_sq = max(0.0, (h_stat - len(CLASSES) + 1) / (n - len(CLASSES)))
        return "kruskal", float(eta_sq), "eta_squared"
    if spec.dtype == "ord":
        rho, _ = sps.spearmanr(series.cat.codes.to_numpy(), y)
        return "spearman", float(rho), "spearman_rho"
    table = pd.crosstab(series, pd.Series(y)).to_numpy()
    chi2, _, _, _ = sps.chi2_contingency(table)
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return "chi2", v, "cramers_v"


def screen_features(
    df: pd.DataFrame, y: np.ndarray, n_perm: int, n_boot: int
) -> list[dict[str, object]]:
    features = [name for name, spec in SCHEMA.items() if spec.role == "feature"]
    results: list[dict[str, object]] = []
    for i, name in enumerate(features):
        t0 = time.time()
        spec = SCHEMA[name]
        if spec.dtype == "int":
            values, codes, n_levels = df[name].to_numpy(dtype=float), None, N_UNIVARIATE_BINS
        else:
            values = None
            codes = df[name].cat.codes.to_numpy()
            n_levels = len(df[name].cat.categories)
            level_counts = np.bincount(codes, minlength=n_levels)
            if (level_counts < MIN_CATEGORY_N).any():  # never true here; guard anyway
                raise RuntimeError(f"{name}: category below MIN_CATEGORY_N — implement merging")

        d = oof_row_deltas(values, codes, n_levels, y)
        delta = float(d.mean())

        rng_boot = np.random.default_rng([SEED, 1, i])
        idx = rng_boot.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
        ci_low, ci_high = (float(q) for q in np.percentile(boot, [2.5, 97.5]))

        rng_perm = np.random.default_rng([SEED, 2, i])
        null = np.empty(n_perm)
        for p in range(n_perm):
            y_perm = rng_perm.permutation(y)
            null[p] = oof_row_deltas(values, codes, n_levels, y_perm).mean()
        perm_p = float((null >= delta).mean())

        test, size, size_name = native_effect(name, df[name], y)
        results.append(
            {
                "feature": name,
                "delta_logloss": delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "permutation_p": perm_p,
                "null_mean": float(null.mean()),
                "null_p95": float(np.percentile(null, 95)),
                "n": int(len(y)),
                "native_test": test,
                "native_effect_size": size,
                "native_effect_name": size_name,
            }
        )
        print(
            f"  [{i + 1:2d}/18] {name:<35s} delta={delta:+.5f} "
            f"perm_p={perm_p:.3f}  ({time.time() - t0:.1f}s)",
            flush=True,
        )

    raw_ps = [float(r["permutation_p"]) for r in results]  # type: ignore[arg-type]
    _, adj_ps, _, _ = multipletests(raw_ps, method="fdr_bh")
    for r, p_adj in zip(results, adj_ps, strict=True):
        r["permutation_p_adj"] = float(p_adj)
        if p_adj < ALPHA and float(r["delta_ci_low"]) > DELTA_NEGLIGIBLE:  # type: ignore[arg-type]
            r["verdict"] = "signal"
        elif (
            float(r["permutation_p"]) > NOISE_PERMUTATION_P  # type: ignore[arg-type]
            and float(r["delta_ci_high"]) < DELTA_NEGLIGIBLE  # type: ignore[arg-type]
        ):
            r["verdict"] = "noise"
        else:
            r["verdict"] = "inconclusive"
    results.sort(key=lambda r: float(r["delta_logloss"]), reverse=True)  # type: ignore[arg-type]
    return results


# ---------------------------------------------------------------- step 3 ---


def build_design(df_tr: pd.DataFrame, df_va: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """M2-style encoding, fitted on the fold-train part only (scaler params)."""
    cols_tr: list[np.ndarray] = []
    cols_va: list[np.ndarray] = []
    for name, spec in SCHEMA.items():
        if spec.role != "feature":
            continue
        if spec.dtype == "int":
            x_tr = df_tr[name].to_numpy(dtype=float)
            x_va = df_va[name].to_numpy(dtype=float)
            mu, sd = x_tr.mean(), x_tr.std()
            sd = sd if sd > 0 else 1.0
            cols_tr.append((x_tr - mu) / sd)
            cols_va.append((x_va - mu) / sd)
        elif spec.dtype == "ord":
            cols_tr.append(df_tr[name].cat.codes.to_numpy(dtype=float))
            cols_va.append(df_va[name].cat.codes.to_numpy(dtype=float))
        else:  # cat / bin: one-hot over the declared categories, drop first
            c_tr = df_tr[name].cat.codes.to_numpy()
            c_va = df_va[name].cat.codes.to_numpy()
            for level in range(1, len(df_tr[name].cat.categories)):
                cols_tr.append((c_tr == level).astype(float))
                cols_va.append((c_va == level).astype(float))
    return np.column_stack(cols_tr), np.column_stack(cols_va)


def cv_model(
    df: pd.DataFrame, y: np.ndarray, kind: str
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Per-fold CV metrics for one model kind: 'prior', 'stratified', 'ordinal', 'hgb'."""
    per_fold: list[dict[str, float]] = []
    notes: list[str] = []
    for k, (tr, va) in enumerate(folds(y)):
        if kind in ("prior", "stratified"):
            freqs = np.bincount(y[tr], minlength=3) / len(tr)
            if kind == "prior":
                proba = np.tile(freqs, (len(va), 1))
            else:
                rng = np.random.default_rng([SEED, 3, k])
                draws = rng.choice(CLASSES, size=len(va), p=freqs)
                proba = np.eye(3)[draws]
        else:
            x_tr, x_va = build_design(df.iloc[tr], df.iloc[va])
            if kind == "ordinal":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = OrderedModel(y[tr], x_tr, distr="logit")
                    res = model.fit(method="bfgs", maxiter=500, disp=False)
                if not res.mle_retvals.get("converged", False):
                    notes.append(f"fold {k}: ordinal logit did not converge")
                proba = np.asarray(res.predict(x_va))
            else:
                clf = HistGradientBoostingClassifier(random_state=SEED)
                clf.fit(x_tr, y[tr])
                proba = clf.predict_proba(x_va)
        per_fold.append(metric_block(y[va], proba))

    summary = {
        metric: {
            "mean": float(np.mean([f[metric] for f in per_fold])),
            "std": float(np.std([f[metric] for f in per_fold])),
        }
        for metric in per_fold[0]
    }
    return summary, notes


# ---------------------------------------------------------------- step 4 ---


def model_permutation_test(
    df: pd.DataFrame, y: np.ndarray, observed_logloss: float, n_perm_model: int
) -> dict[str, float]:
    """Null distribution of HGB CV log loss under target permutation.

    Runs on the gradient boosting model: it is the only fitted model that can
    represent interactions, which is exactly the case this diagnostic guards
    against (SPEC M3 known limitation). Diagnostic only — never alters the
    dataset verdict.
    """
    rng = np.random.default_rng([SEED, 4])
    null = np.empty(n_perm_model)
    for p in range(n_perm_model):
        y_perm = rng.permutation(y)
        losses = []
        for tr, va in folds(y_perm):
            x_tr, x_va = build_design(df.iloc[tr], df.iloc[va])
            clf = HistGradientBoostingClassifier(random_state=SEED)
            clf.fit(x_tr, y_perm[tr])
            losses.append(row_log_loss(clf.predict_proba(x_va), y_perm[va]).mean())
        null[p] = float(np.mean(losses))
        if (p + 1) % 20 == 0:
            print(f"  model permutation {p + 1}/{n_perm_model}", flush=True)
    return {
        "observed_cv_logloss": observed_logloss,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p05": float(np.percentile(null, 5)),
        "p_value": float((null <= observed_logloss).mean()),
    }


# ------------------------------------------------------------------ main ---


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny iteration counts")
    args = parser.parse_args()
    n_perm = 20 if args.smoke else N_PERM
    n_boot = 200 if args.smoke else N_BOOT
    n_perm_model = 5 if args.smoke else N_PERM_MODEL

    t_start = time.time()
    raw = load_raw()
    report = validate(raw)
    if not report.is_valid:
        print(f"Validation failed: {report}", flush=True)
        return 1
    df_all = prepare(raw)
    train_df, _test_df = split(df_all)  # test set: loaded, split off, never used here
    train_df = train_df.reset_index(drop=True)
    y = train_df[TARGET].cat.codes.to_numpy()
    x_df = train_df.drop(columns=[TARGET])
    print(f"Train split: {len(train_df)} rows; test split untouched.", flush=True)

    print("\n[1/4] Dummy baselines (OOF on train)", flush=True)
    baselines = {kind: cv_model(x_df, y, kind)[0] for kind in ("prior", "stratified")}
    for kind, summary in baselines.items():
        print(f"  {kind:<11s} " + "  ".join(f"{m}={v['mean']:.4f}" for m, v in summary.items()))

    print(f"\n[2/4] Common statistic per feature ({n_perm} permutations)", flush=True)
    effects = screen_features(x_df, y, n_perm=n_perm, n_boot=n_boot)
    n_signal = sum(1 for r in effects if r["verdict"] == "signal")

    print("\n[3/4] Models under CV", flush=True)
    models: dict[str, dict[str, dict[str, float]]] = {}
    model_notes: dict[str, list[str]] = {}
    for kind in ("ordinal", "hgb"):
        models[kind], model_notes[kind] = cv_model(x_df, y, kind)
        print(
            f"  {kind:<8s} "
            + "  ".join(f"{m}={v['mean']:.4f}±{v['std']:.4f}" for m, v in models[kind].items())
        )

    base_ll = baselines["prior"]["log_loss"]
    beats = {
        kind: (base_ll["mean"] - models[kind]["log_loss"]["mean"])
        > BASELINE_SD_MULTIPLIER * models[kind]["log_loss"]["std"]
        for kind in models
    }

    print(f"\n[4/4] Model-level permutation test ({n_perm_model} shuffles, HGB)", flush=True)
    perm_test = model_permutation_test(
        x_df, y, models["hgb"]["log_loss"]["mean"], n_perm_model=n_perm_model
    )

    dataset_verdict = "signal_present" if n_signal > 0 else "no_detectable_signal"

    out = {
        "protocol": "verify-signal (Phase 0 spike)",
        "n_train": int(len(train_df)),
        "seed": SEED,
        "n_perm": n_perm,
        "n_boot": n_boot,
        "n_perm_model": n_perm_model,
        "baselines": baselines,
        "effects": effects,
        "n_features_tested": len(effects),
        "n_signal": n_signal,
        "models": models,
        "model_notes": model_notes,
        "beats_baseline_on_logloss": beats,
        "model_permutation_test": perm_test,
        "dataset_verdict": dataset_verdict,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    path = ARTIFACTS_DIR / "signal_report.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nDataset verdict: {dataset_verdict}  (n_signal={n_signal}/18)")
    print(f"Report written to {path}  ({out['runtime_seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
