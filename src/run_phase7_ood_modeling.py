#!/usr/bin/env python3
"""
Phase 7.9 — Real-data held-out-domain OOD modeling

Input:
  results/experiment_7/plant_level_dataset/phase7_plant_level_dataset.csv

Protocol:
- 216 individual plants, 18 experiment×replicate×treatment domains.
- Primary target: total_weight_g.
- A0 only: no synthetic/oracle augmentation.
- Leave-one-domain-out (LODO): 17 domains train, 1 unseen domain OOD test.
- ID estimate inside each outer fold: GroupKFold on the 17 training domains.
- Preprocessing is fitted inside training folds only.
- Domain IDs / experiment / replicate / treatment / provenance are NOT predictors.
- Environmental predictors are domain-level history summaries.
- Shift severity V is the standardized Euclidean distance between the held-out
  domain's environmental feature centroid and the outer-training centroid,
  using scaler parameters fitted on outer-training plants only.
- Models:
    M1 = small MLPRegressor (16, 8)
    M0 = medium MLPRegressor (32, 16)
    M2 = large MLPRegressor (64, 32, 16)
  These mirror the relative capacity ordering used in the synthetic phases,
  while remaining appropriate to the small real dataset.

Outputs:
  results/experiment_7/ood_modeling/
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "experiment_7" / "plant_level_dataset" / "phase7_plant_level_dataset.csv"
OUT = ROOT / "results" / "experiment_7" / "ood_modeling"

FOLD_CSV = OUT / "phase7_lodo_fold_results.csv"
PRED_CSV = OUT / "phase7_lodo_predictions.csv"
SUMMARY_CSV = OUT / "phase7_model_summary.csv"
SEVERITY_CSV = OUT / "phase7_shift_severity_results.csv"
REPORT = OUT / "phase7_ood_modeling_report.txt"
MANIFEST = OUT / "phase7_ood_modeling_manifest.json"

EXPECTED_ROWS = 216
EXPECTED_DOMAINS = 18
RANDOM_STATE = 20260917
INNER_SPLITS = 5

MODELS = {
    "M1": (16, 8),
    "M0": (32, 16),
    "M2": (64, 32, 16),
}


def safe_r2(y_true, y_pred) -> float:
    if len(y_true) < 2:
        return math.nan
    return float(r2_score(y_true, y_pred))


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": safe_r2(y_true, y_pred),
    }


def make_model(hidden_layers: tuple[int, ...], seed: int) -> Pipeline:
    # Median imputation + scaling are learned only from each training fold.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="lbfgs",
            alpha=1e-2,
            max_iter=3000,
            random_state=seed,
        )),
    ])


def domain_key(df: pd.DataFrame) -> pd.Series:
    return df["experiment"].astype(str) + "__" + df["domain"].astype(str)


def choose_features(df: pd.DataFrame) -> list[str]:
    """
    Use only environmental summary statistics frozen in P7.7/P7.8.

    Exclude *_n because measurement availability/count is metadata rather than
    an environmental exposure. Exclude date/window/count/provenance fields.
    """
    allowed_suffixes = (
        "_mean", "_std", "_min", "_max", "_q25", "_median", "_q75", "_slope"
    )
    features = [
        c for c in df.columns
        if c.startswith("env_") and c.endswith(allowed_suffixes)
    ]
    return sorted(features)


def fit_feature_transform(X_train: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_train)
    scaler = StandardScaler()
    scaler.fit(X_imp)
    return imp, scaler


def shift_severity(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> float:
    """
    Standardized centroid distance, fitted using outer-training data only.

    Divide Euclidean norm by sqrt(p) so V is interpretable as RMS standardized
    feature displacement and does not grow mechanically with feature count.
    """
    imp, scaler = fit_feature_transform(X_train)
    tr = scaler.transform(imp.transform(X_train))
    te = scaler.transform(imp.transform(X_test))
    delta = np.nanmean(te, axis=0) - np.nanmean(tr, axis=0)
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


def correlation(x, y, method: str) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan, math.nan
    xx, yy = x[mask], y[mask]
    if np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return math.nan, math.nan
    if method == "pearson":
        r, p = pearsonr(xx, yy)
    else:
        r, p = spearmanr(xx, yy)
    return float(r), float(p)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    notes: list[str] = []

    if not INPUT.exists():
        print(f"ERROR: missing validated P7.8 dataset: {INPUT}")
        return 1

    df = pd.read_csv(INPUT)
    features = choose_features(df)

    required = {"experiment", "domain", "plant_key", "total_weight_g"}
    missing_required = sorted(required - set(df.columns))
    if missing_required:
        errors.append(f"Missing required columns: {missing_required}")

    if len(df) != EXPECTED_ROWS:
        errors.append(f"Input rows={len(df)}, expected {EXPECTED_ROWS}")

    if not missing_required:
        df["domain_key"] = domain_key(df)
        n_domains = int(df["domain_key"].nunique())
        if n_domains != EXPECTED_DOMAINS:
            errors.append(f"Input domains={n_domains}, expected {EXPECTED_DOMAINS}")
    else:
        n_domains = 0

    if not features:
        errors.append("No frozen environmental predictor columns found.")

    if "total_weight_g" in df and df["total_weight_g"].isna().any():
        errors.append("Primary target contains missing values.")

    all_missing = [c for c in features if df[c].isna().all()]
    if all_missing:
        errors.append(f"All-missing predictors: {all_missing}")

    if errors:
        REPORT.write_text(
            "PHASE 7.9 — REAL-DATA HELD-OUT-DOMAIN OOD MODELING\n"
            + "=" * 112 + "\n"
            + "PRE-FLIGHT FAILED\n"
            + "\n".join(f"ERROR: {e}" for e in errors) + "\n",
            encoding="utf-8",
        )
        print(REPORT.read_text(encoding="utf-8"))
        return 1

    # The feature table contains repeated domain-level histories across 12
    # plants/domain. This is intentional. Domain-wise splitting prevents those
    # histories from appearing in both outer train and OOD test.
    y = df["total_weight_g"].to_numpy(dtype=float)
    domains = sorted(df["domain_key"].unique())

    fold_rows = []
    prediction_rows = []

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for fold_idx, heldout in enumerate(domains, start=1):
        test_mask = df["domain_key"].eq(heldout).to_numpy()
        train_mask = ~test_mask

        train_df = df.loc[train_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        X_train = train_df[features]
        y_train = train_df["total_weight_g"].to_numpy(dtype=float)
        g_train = train_df["domain_key"].to_numpy()
        X_test = test_df[features]
        y_test = test_df["total_weight_g"].to_numpy(dtype=float)

        severity = shift_severity(X_train, X_test)

        # Same held-out domain has the same V for all model configurations.
        for model_idx, (model_name, layers) in enumerate(MODELS.items()):
            seed_base = RANDOM_STATE + fold_idx * 100 + model_idx * 10

            # ID estimate is group-aware CV using only outer-training domains.
            gkf = GroupKFold(n_splits=INNER_SPLITS)
            id_true_parts = []
            id_pred_parts = []

            for inner_idx, (itr, iva) in enumerate(
                gkf.split(X_train, y_train, groups=g_train), start=1
            ):
                model = make_model(layers, seed_base + inner_idx)
                model.fit(X_train.iloc[itr], y_train[itr])
                pred = model.predict(X_train.iloc[iva])
                id_true_parts.append(y_train[iva])
                id_pred_parts.append(pred)

            id_true = np.concatenate(id_true_parts)
            id_pred = np.concatenate(id_pred_parts)
            idm = metrics(id_true, id_pred)

            # Final outer model: all 17 training domains -> one unseen domain.
            outer_model = make_model(layers, seed_base + 99)
            outer_model.fit(X_train, y_train)
            ood_pred = outer_model.predict(X_test)
            oodm = metrics(y_test, ood_pred)

            fold_rows.append({
                "fold": fold_idx,
                "heldout_domain": heldout,
                "heldout_experiment": str(test_df["experiment"].iloc[0]),
                "heldout_domain_label": str(test_df["domain"].iloc[0]),
                "model": model_name,
                "hidden_layers": str(layers),
                "n_train_plants": len(train_df),
                "n_test_plants": len(test_df),
                "n_train_domains": train_df["domain_key"].nunique(),
                "shift_severity_v": severity,
                "id_mae": idm["mae"],
                "id_rmse": idm["rmse"],
                "id_r2": idm["r2"],
                "ood_mae": oodm["mae"],
                "ood_rmse": oodm["rmse"],
                "ood_r2": oodm["r2"],
                "delta_mae": oodm["mae"] - idm["mae"],
                "delta_rmse": oodm["rmse"] - idm["rmse"],
            })

            for i, row in test_df.iterrows():
                prediction_rows.append({
                    "fold": fold_idx,
                    "heldout_domain": heldout,
                    "model": model_name,
                    "plant_key": row["plant_key"],
                    "y_true": float(y_test[i]),
                    "y_pred": float(ood_pred[i]),
                    "residual": float(y_test[i] - ood_pred[i]),
                    "abs_error": float(abs(y_test[i] - ood_pred[i])),
                    "shift_severity_v": severity,
                })

    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(prediction_rows)
    folds.to_csv(FOLD_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)

    # Descriptive model summaries across the 18 outer domains.
    summaries = []
    for model_name, g in folds.groupby("model", sort=False):
        summaries.append({
            "model": model_name,
            "domains": len(g),
            "mean_id_mae": g["id_mae"].mean(),
            "mean_ood_mae": g["ood_mae"].mean(),
            "mean_delta_mae": g["delta_mae"].mean(),
            "median_delta_mae": g["delta_mae"].median(),
            "domains_ood_mae_gt_id_mae": int((g["delta_mae"] > 0).sum()),
            "mean_id_rmse": g["id_rmse"].mean(),
            "mean_ood_rmse": g["ood_rmse"].mean(),
            "mean_delta_rmse": g["delta_rmse"].mean(),
            "mean_ood_r2": g["ood_r2"].mean(),
        })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    # H7.2 descriptive/association evidence: V vs degradation by model.
    sev_rows = []
    for model_name, g in folds.groupby("model", sort=False):
        for outcome in ("delta_mae", "delta_rmse", "ood_mae", "ood_rmse"):
            pr, pp = correlation(g["shift_severity_v"], g[outcome], "pearson")
            sr, sp = correlation(g["shift_severity_v"], g[outcome], "spearman")
            sev_rows.append({
                "model": model_name,
                "outcome": outcome,
                "n_domains": len(g),
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
            })
    sev_df = pd.DataFrame(sev_rows)
    sev_df.to_csv(SEVERITY_CSV, index=False)

    # Conservative status: P7.9 is an analysis stage, not a universal proof.
    finite_primary = (
        np.isfinite(folds["ood_mae"]).all()
        and np.isfinite(folds["id_mae"]).all()
        and np.isfinite(folds["shift_severity_v"]).all()
    )
    expected_fold_rows = EXPECTED_DOMAINS * len(MODELS)
    if len(folds) != expected_fold_rows:
        errors.append(f"Fold-result rows={len(folds)}, expected {expected_fold_rows}.")
    if len(preds) != EXPECTED_ROWS * len(MODELS):
        errors.append(
            f"Prediction rows={len(preds)}, expected {EXPECTED_ROWS * len(MODELS)}."
        )
    if not finite_primary:
        errors.append("Non-finite value detected in primary MAE/severity results.")

    verdict = "PASS" if not errors else "REVIEW REQUIRED"

    lines = [
        "PHASE 7.9 — REAL-DATA HELD-OUT-DOMAIN OOD MODELING",
        "=" * 112,
        f"Input: {INPUT}",
        f"Input plants: {len(df)}",
        f"Input domains: {n_domains}",
        f"Environmental predictors: {len(features)}",
        f"Models: {', '.join(MODELS)}",
        "",
        "PROTOCOL",
        "-" * 112,
        "Outer evaluation: leave one experiment×replicate×treatment domain out (18 outer folds)",
        "Outer training set per fold: 17 domains / 204 plants",
        "Outer OOD test set per fold: 1 unseen domain / 12 plants",
        f"ID estimate: {INNER_SPLITS}-fold GroupKFold using only the 17 outer-training domains",
        "Preprocessing: median imputation + standardization fitted within training folds only",
        "Domain/experiment/replicate/treatment identifiers are NOT model predictors",
        "Augmentation: A0 only; no synthetic oracle relabeling",
        "Shift severity V: RMS standardized centroid distance, fitted from outer-training data only",
        "",
        "INTEGRITY",
        "-" * 112,
        f"Outer fold/model result rows: {len(folds)} / expected {expected_fold_rows}",
        f"OOD plant predictions: {len(preds)} / expected {EXPECTED_ROWS * len(MODELS)}",
        f"Finite primary MAE/severity results: {'YES' if finite_primary else 'NO'}",
        "",
        "MODEL SUMMARY ACROSS 18 HELD-OUT DOMAINS",
        "-" * 112,
    ]

    for _, r in summary_df.iterrows():
        lines.extend([
            f"{r['model']}:",
            f"  mean ID MAE={r['mean_id_mae']:.6f}",
            f"  mean OOD MAE={r['mean_ood_mae']:.6f}",
            f"  mean ΔMAE (OOD-ID)={r['mean_delta_mae']:.6f}",
            f"  median ΔMAE={r['median_delta_mae']:.6f}",
            f"  domains with OOD MAE > ID MAE={int(r['domains_ood_mae_gt_id_mae'])}/18",
            f"  mean ID RMSE={r['mean_id_rmse']:.6f}",
            f"  mean OOD RMSE={r['mean_ood_rmse']:.6f}",
            f"  mean ΔRMSE={r['mean_delta_rmse']:.6f}",
            f"  mean OOD R²={r['mean_ood_r2']:.6f}",
        ])

    lines.extend([
        "",
        "H7.2 — SHIFT-SEVERITY ASSOCIATION (DESCRIPTIVE DOMAIN-LEVEL TESTS)",
        "-" * 112,
        "Positive correlation means larger measured environmental shift is associated with larger error/degradation.",
    ])

    for _, r in sev_df[sev_df["outcome"].eq("delta_mae")].iterrows():
        lines.append(
            f"{r['model']} V vs ΔMAE: "
            f"Pearson r={r['pearson_r']:.6f}, p={r['pearson_p']:.6g}; "
            f"Spearman rho={r['spearman_rho']:.6f}, p={r['spearman_p']:.6g}"
        )

    lines.extend([
        "",
        "INTERPRETATION GUARDRAILS",
        "-" * 112,
        "This is external real-data validation under observed/experimental hydroponic domains.",
        "It is not equivalent to the controlled causal synthetic shift experiments.",
        "The 12 plants within a domain share the same environmental-history predictors.",
        "Therefore domain-level patterns and uncertainty must not be interpreted as 216 independent environmental histories.",
        "R² for individual 12-plant held-out domains can be unstable and is secondary to MAE/RMSE.",
        "No universal claim is made from a single real dataset.",
        "",
        "VALIDATION ERRORS",
        "-" * 112,
        f"Error count: {len(errors)}",
    ])
    lines.extend([f"ERROR: {e}" for e in errors] or ["None"])

    lines.extend([
        "",
        "FINAL VERDICT",
        "=" * 112,
        verdict,
        "",
        "RESEARCH STATUS",
        "-" * 112,
        "P7.8 validated plant-level dataset used: YES",
        "Held-out-domain OOD protocol executed: YES" if verdict == "PASS" else "Held-out-domain OOD protocol executed: REVIEW",
        "Synthetic oracle augmentation applied: NO",
        "Raw archive modified: NO",
        "Universal/causal real-data claim authorized: NO",
        "Next if PASS: formal H7.1/H7.2/H7.3 statistical analysis and Phase-7 synthesis.",
    ])

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "phase": "P7.9",
        "input": str(INPUT),
        "rows": len(df),
        "domains": n_domains,
        "features": features,
        "models": {k: list(v) for k, v in MODELS.items()},
        "random_state": RANDOM_STATE,
        "outer_protocol": "leave-one-domain-out",
        "inner_protocol": f"{INNER_SPLITS}-fold GroupKFold on outer-training domains",
        "shift_severity": "RMS standardized centroid distance; training-fit imputer/scaler",
        "augmentation": "A0 only",
        "fold_result_rows": len(folds),
        "prediction_rows": len(preds),
        "errors": errors,
        "verdict": verdict,
        "raw_archive_modified": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(REPORT.read_text(encoding="utf-8"))
    print(f"Fold results: {FOLD_CSV}")
    print(f"Predictions: {PRED_CSV}")
    print(f"Report: {REPORT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
