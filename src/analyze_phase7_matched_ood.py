#!/usr/bin/env python3
"""
Phase 7.10 — Matched domain-level OOD degradation and severity diagnostics

This stage corrects the scientific comparability problem identified after P7.9.

Instead of comparing:
  inner-CV ID error on heterogeneous validation domains
against:
  one held-out-domain OOD error,

P7.10 creates a MATCHED reference for every held-out domain:
  1. Train the same model on the other 17 domains.
  2. Predict the 12 plants in the held-out domain.
  3. Compute OOD MAE/RMSE.
  4. Compute a domain-matched naive reference using ONLY outer-training target
     information: the outer-training mean target predicts every plant in the
     held-out domain.
  5. Define excess error relative to that matched reference:
       excess_mae  = model_ood_mae  - matched_reference_mae
       excess_rmse = model_ood_rmse - matched_reference_rmse

This does NOT redefine the held-out domain as "ID". It removes the invalid
cross-validation-vs-single-domain subtraction and provides a matched
domain-level benchmark for interpretation.

It also:
- reconstructs shift severity V exactly from the validated P7.8 feature table;
- diagnoses Pearson-vs-Spearman disagreement;
- reports leave-one-domain-out influence for V/error correlations;
- fits domain-level OLS models for H7.2/H7.3 with HC3 robust SE;
- treats the 18 domains, not 216 plants, as the independent units for
  severity/inference;
- does not make causal or universal claims.

Inputs:
  results/experiment_7/plant_level_dataset/phase7_plant_level_dataset.csv
  results/experiment_7/ood_modeling/phase7_lodo_fold_results.csv
  results/experiment_7/ood_modeling/phase7_lodo_predictions.csv

Outputs:
  results/experiment_7/matched_ood_analysis/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "experiment_7" / "plant_level_dataset" / "phase7_plant_level_dataset.csv"
FOLDS = ROOT / "results" / "experiment_7" / "ood_modeling" / "phase7_lodo_fold_results.csv"
PREDS = ROOT / "results" / "experiment_7" / "ood_modeling" / "phase7_lodo_predictions.csv"

OUT = ROOT / "results" / "experiment_7" / "matched_ood_analysis"
DOMAIN_CSV = OUT / "phase7_matched_domain_results.csv"
INFLUENCE_CSV = OUT / "phase7_severity_influence_diagnostics.csv"
OLS_CSV = OUT / "phase7_h73_ols_coefficients.csv"
REPORT = OUT / "phase7_matched_ood_analysis_report.txt"
MANIFEST = OUT / "phase7_matched_ood_analysis_manifest.json"

EXPECTED_PLANTS = 216
EXPECTED_DOMAINS = 18
MODELS = ("M1", "M0", "M2")


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def choose_features(df: pd.DataFrame) -> list[str]:
    suffixes = ("_mean", "_std", "_min", "_max", "_q25", "_median", "_q75", "_slope")
    return sorted(
        c for c in df.columns
        if c.startswith("env_") and c.endswith(suffixes)
    )


def domain_key(df: pd.DataFrame) -> pd.Series:
    return df["experiment"].astype(str) + "__" + df["domain"].astype(str)


def severity_from_training(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> float:
    imp = SimpleImputer(strategy="median")
    xtr = imp.fit_transform(train[features])
    xte = imp.transform(test[features])

    scaler = StandardScaler()
    ztr = scaler.fit_transform(xtr)
    zte = scaler.transform(xte)

    delta = zte.mean(axis=0) - ztr.mean(axis=0)
    return float(np.linalg.norm(delta) / np.sqrt(len(delta)))


def corr(x, y, kind: str) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return math.nan, math.nan
    if kind == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p)


def safe_wilcoxon(values) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.allclose(x, 0):
        return math.nan, math.nan
    stat, p = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for path in (DATASET, FOLDS, PREDS):
        if not path.exists():
            errors.append(f"Missing required input: {path}")

    if errors:
        REPORT.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(REPORT.read_text(encoding="utf-8"))
        return 1

    df = pd.read_csv(DATASET)
    folds = pd.read_csv(FOLDS)
    preds = pd.read_csv(PREDS)

    df["domain_key"] = domain_key(df)
    features = choose_features(df)

    if len(df) != EXPECTED_PLANTS:
        errors.append(f"Dataset rows={len(df)}, expected {EXPECTED_PLANTS}.")
    if df["domain_key"].nunique() != EXPECTED_DOMAINS:
        errors.append(f"Domains={df['domain_key'].nunique()}, expected {EXPECTED_DOMAINS}.")
    if len(folds) != EXPECTED_DOMAINS * len(MODELS):
        errors.append(f"P7.9 fold rows={len(folds)}, expected 54.")
    if len(preds) != EXPECTED_PLANTS * len(MODELS):
        errors.append(f"P7.9 prediction rows={len(preds)}, expected 648.")
    if not features:
        errors.append("No environmental summary predictors found.")

    # Build one matched row per held-out domain × model.
    rows = []
    for _, fr in folds.iterrows():
        held = str(fr["heldout_domain"])
        model = str(fr["model"])

        train = df.loc[df["domain_key"] != held].copy()
        test = df.loc[df["domain_key"] == held].copy()

        pp = preds.loc[
            (preds["heldout_domain"].astype(str) == held)
            & (preds["model"].astype(str) == model)
        ].copy()

        if len(test) != 12:
            errors.append(f"{held}: test plants={len(test)}, expected 12.")
            continue
        if len(pp) != 12:
            errors.append(f"{held}/{model}: predictions={len(pp)}, expected 12.")
            continue

        # Align by plant key rather than row order.
        joined = test[["plant_key", "total_weight_g"]].merge(
            pp[["plant_key", "y_pred"]],
            on="plant_key",
            how="inner",
            validate="one_to_one",
        )
        if len(joined) != 12:
            errors.append(f"{held}/{model}: plant-key prediction alignment failed.")
            continue

        y_test = joined["total_weight_g"].to_numpy(dtype=float)
        y_pred = joined["y_pred"].to_numpy(dtype=float)

        ood_mae = float(mean_absolute_error(y_test, y_pred))
        ood_rmse = rmse(y_test, y_pred)

        # Matched reference: uses training targets only, then evaluated on the
        # exact same held-out 12 plants.
        train_mean = float(train["total_weight_g"].mean())
        ref_pred = np.full(len(y_test), train_mean, dtype=float)
        ref_mae = float(mean_absolute_error(y_test, ref_pred))
        ref_rmse = rmse(y_test, ref_pred)

        v = severity_from_training(train, test, features)

        rows.append({
            "heldout_domain": held,
            "experiment": str(test["experiment"].iloc[0]),
            "domain": str(test["domain"].iloc[0]),
            "model": model,
            "n_test_plants": len(test),
            "shift_severity_v": v,
            "ood_mae": ood_mae,
            "ood_rmse": ood_rmse,
            "matched_reference_mae": ref_mae,
            "matched_reference_rmse": ref_rmse,
            "excess_mae": ood_mae - ref_mae,
            "excess_rmse": ood_rmse - ref_rmse,
            "model_minus_reference_mae": ood_mae - ref_mae,
            "model_minus_reference_rmse": ood_rmse - ref_rmse,
            "p79_id_mae_for_diagnostic_only": float(fr["id_mae"]),
            "p79_delta_mae_for_diagnostic_only": float(fr["delta_mae"]),
        })

    matched = pd.DataFrame(rows)
    matched.to_csv(DOMAIN_CSV, index=False)

    # Check reconstructed severity against P7.9.
    merged_v = matched.merge(
        folds[["heldout_domain", "model", "shift_severity_v"]],
        on=["heldout_domain", "model"],
        suffixes=("_p710", "_p79"),
        validate="one_to_one",
    )
    max_v_diff = float(
        np.max(np.abs(
            merged_v["shift_severity_v_p710"] - merged_v["shift_severity_v_p79"]
        ))
    ) if len(merged_v) else math.nan
    if not np.isfinite(max_v_diff) or max_v_diff > 1e-10:
        errors.append(f"Reconstructed V differs from P7.9; max abs difference={max_v_diff}")

    # H7.1 is NOT tested as "OOD > ID" here because a scientifically matched
    # ID observation for the same unseen domain does not exist. Instead report
    # performance relative to the matched training-only mean reference.
    model_summaries = []
    for model in MODELS:
        g = matched.loc[matched["model"] == model].copy()
        w_stat, w_p = safe_wilcoxon(g["excess_mae"])
        model_summaries.append({
            "model": model,
            "domains": len(g),
            "mean_ood_mae": float(g["ood_mae"].mean()),
            "mean_reference_mae": float(g["matched_reference_mae"].mean()),
            "mean_excess_mae": float(g["excess_mae"].mean()),
            "median_excess_mae": float(g["excess_mae"].median()),
            "domains_model_worse_than_reference": int((g["excess_mae"] > 0).sum()),
            "wilcoxon_excess_mae_stat": w_stat,
            "wilcoxon_excess_mae_p": w_p,
        })

    # H7.2: severity associations using domain-level units.
    h72 = []
    influence_rows = []
    for model in MODELS:
        g = matched.loc[matched["model"] == model].sort_values("heldout_domain").reset_index(drop=True)

        for outcome in ("ood_mae", "ood_rmse", "excess_mae", "excess_rmse"):
            pr, pp = corr(g["shift_severity_v"], g[outcome], "pearson")
            sr, sp = corr(g["shift_severity_v"], g[outcome], "spearman")
            h72.append({
                "model": model,
                "outcome": outcome,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_rho": sr,
                "spearman_p": sp,
            })

            # Leave-one-domain-out influence: if Pearson collapses when one
            # domain is removed, the association is outlier-sensitive.
            for i in range(len(g)):
                gg = g.drop(index=i)
                lr, lp = corr(gg["shift_severity_v"], gg[outcome], "pearson")
                influence_rows.append({
                    "model": model,
                    "outcome": outcome,
                    "removed_domain": g.loc[i, "heldout_domain"],
                    "removed_v": float(g.loc[i, "shift_severity_v"]),
                    "removed_outcome": float(g.loc[i, outcome]),
                    "pearson_r_without_domain": lr,
                    "pearson_p_without_domain": lp,
                })

    h72_df = pd.DataFrame(h72)
    influence = pd.DataFrame(influence_rows)
    influence.to_csv(INFLUENCE_CSV, index=False)

    # H7.3: domain-level model × severity interaction.
    # There are 54 rows but only 18 unique domains, so HC3 is used and results
    # are explicitly treated as exploratory/small-sample association evidence.
    ols_rows = []
    for outcome in ("ood_mae", "excess_mae"):
        fit = smf.ols(
            f"{outcome} ~ shift_severity_v * C(model, Treatment(reference='M0'))",
            data=matched,
        ).fit(cov_type="HC3")

        for term in fit.params.index:
            ols_rows.append({
                "outcome": outcome,
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error_hc3": float(fit.bse[term]),
                "t": float(fit.tvalues[term]),
                "p": float(fit.pvalues[term]),
                "ci_low": float(fit.conf_int().loc[term, 0]),
                "ci_high": float(fit.conf_int().loc[term, 1]),
                "n_rows": int(fit.nobs),
                "unique_domains": EXPECTED_DOMAINS,
                "r_squared": float(fit.rsquared),
            })

    ols_df = pd.DataFrame(ols_rows)
    ols_df.to_csv(OLS_CSV, index=False)

    # Diagnostic ranges for severity and influential observations.
    domain_v = (
        matched[["heldout_domain", "shift_severity_v"]]
        .drop_duplicates()
        .sort_values("shift_severity_v", ascending=False)
    )
    v_min = float(domain_v["shift_severity_v"].min())
    v_median = float(domain_v["shift_severity_v"].median())
    v_max = float(domain_v["shift_severity_v"].max())
    top_v = domain_v.head(5)

    # Primary H7.2 table for OOD MAE and excess MAE.
    primary_h72 = h72_df[h72_df["outcome"].isin(["ood_mae", "excess_mae"])]

    # Influence ranges.
    influence_summary = []
    for model in MODELS:
        for outcome in ("ood_mae", "excess_mae"):
            z = influence[
                (influence["model"] == model)
                & (influence["outcome"] == outcome)
            ]
            influence_summary.append({
                "model": model,
                "outcome": outcome,
                "min_leave1out_pearson_r": float(z["pearson_r_without_domain"].min()),
                "max_leave1out_pearson_r": float(z["pearson_r_without_domain"].max()),
            })

    if len(matched) != 54:
        errors.append(f"Matched domain/model rows={len(matched)}, expected 54.")
    if matched["heldout_domain"].nunique() != 18:
        errors.append("Matched results do not contain exactly 18 held-out domains.")
    if not np.isfinite(matched[["ood_mae", "matched_reference_mae", "shift_severity_v"]].to_numpy()).all():
        errors.append("Non-finite primary matched metrics detected.")

    verdict = "PASS" if not errors else "REVIEW REQUIRED"

    lines = [
        "PHASE 7.10 — MATCHED DOMAIN-LEVEL OOD DEGRADATION AND SEVERITY DIAGNOSTICS",
        "=" * 118,
        f"Validated plant dataset: {DATASET}",
        f"P7.9 fold results: {FOLDS}",
        f"P7.9 predictions: {PREDS}",
        "",
        "WHY P7.10 WAS REQUIRED",
        "-" * 118,
        "P7.9 correctly executed held-out-domain prediction, but its ΔMAE subtracted an inner GroupKFold ID",
        "estimate pooled across heterogeneous validation domains from a single held-out-domain OOD MAE.",
        "Those two errors are not matched evaluation quantities, so P7.9 ΔMAE is retained for diagnosis only and",
        "is NOT used as the primary H7.1 degradation statistic.",
        "",
        "MATCHED REFERENCE",
        "-" * 118,
        "For each held-out domain, the reference predicts the outer-training mean target on the exact same 12 test plants.",
        "The reference uses no held-out targets during fitting. It is a matched predictive benchmark, NOT an ID observation.",
        "excess_mae = OOD model MAE - matched reference MAE.",
        "Negative excess_mae means the trained model beats the matched training-only mean reference.",
        "Positive excess_mae means the trained model is worse than that reference.",
        "",
        "INTEGRITY",
        "-" * 118,
        f"Plant rows: {len(df)} / expected 216",
        f"Independent environmental domains: {df['domain_key'].nunique()} / expected 18",
        f"Matched domain×model rows: {len(matched)} / expected 54",
        f"Environmental features: {len(features)}",
        f"Maximum |V(P7.10)-V(P7.9)|: {max_v_diff:.12g}",
        "",
        "MATCHED PERFORMANCE BY MODEL",
        "-" * 118,
    ]

    for s in model_summaries:
        lines.extend([
            f"{s['model']}:",
            f"  mean OOD MAE={s['mean_ood_mae']:.6f}",
            f"  mean matched-reference MAE={s['mean_reference_mae']:.6f}",
            f"  mean excess MAE={s['mean_excess_mae']:.6f}",
            f"  median excess MAE={s['median_excess_mae']:.6f}",
            f"  domains model worse than reference={s['domains_model_worse_than_reference']}/18",
            f"  Wilcoxon excess-MAE two-sided p={s['wilcoxon_excess_mae_p']:.6g}",
        ])

    lines.extend([
        "",
        "H7.1 STATUS",
        "-" * 118,
        "The original statement L_OOD > L_ID is NOT directly identifiable from this dataset/protocol because the same",
        "held-out domain cannot simultaneously be unseen OOD and observed ID for a matched model evaluation.",
        "Therefore P7.10 does not manufacture an ID comparator. It reports matched held-out-domain performance against",
        "a training-only reference and leaves H7.1 as not directly tested by the available real-data design.",
        "",
        "H7.2 — SHIFT SEVERITY ASSOCIATIONS",
        "-" * 118,
        f"V range across 18 domains: min={v_min:.6f}, median={v_median:.6f}, max={v_max:.6f}",
    ])

    for _, r in primary_h72.iterrows():
        lines.append(
            f"{r['model']} | V vs {r['outcome']}: "
            f"Pearson r={r['pearson_r']:.6f}, p={r['pearson_p']:.6g}; "
            f"Spearman rho={r['spearman_rho']:.6f}, p={r['spearman_p']:.6g}"
        )

    lines.extend([
        "",
        "HIGHEST-SEVERITY DOMAINS",
        "-" * 118,
    ])
    for _, r in top_v.iterrows():
        lines.append(f"{r['heldout_domain']}: V={r['shift_severity_v']:.6f}")

    lines.extend([
        "",
        "PEARSON LEAVE-ONE-DOMAIN INFLUENCE RANGES",
        "-" * 118,
        "Large changes across these ranges indicate sensitivity to individual domains.",
    ])
    for s in influence_summary:
        lines.append(
            f"{s['model']} | {s['outcome']}: "
            f"r_without_one_domain range "
            f"[{s['min_leave1out_pearson_r']:.6f}, {s['max_leave1out_pearson_r']:.6f}]"
        )

    lines.extend([
        "",
        "H7.3 — MODEL × SEVERITY OLS (HC3; M0 REFERENCE)",
        "-" * 118,
        "Interaction terms test whether the slope with V differs from M0. These are exploratory because only 18",
        "unique environmental domains underlie the 54 model-domain rows.",
    ])
    for _, r in ols_df.iterrows():
        if ":shift_severity_v" in r["term"] or "shift_severity_v:C(" in r["term"]:
            lines.append(
                f"{r['outcome']} | {r['term']}: "
                f"b={r['estimate']:.6f}, HC3 SE={r['std_error_hc3']:.6f}, p={r['p']:.6g}, "
                f"95% CI=[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
            )

    # Statsmodels may order interaction names as C(...):shift_severity_v.
    if not any(":shift_severity_v" in t for t in ols_df["term"]):
        for _, r in ols_df.iterrows():
            if ":" in r["term"]:
                lines.append(
                    f"{r['outcome']} | {r['term']}: "
                    f"b={r['estimate']:.6f}, HC3 SE={r['std_error_hc3']:.6f}, p={r['p']:.6g}, "
                    f"95% CI=[{r['ci_low']:.6f}, {r['ci_high']:.6f}]"
                )

    lines.extend([
        "",
        "INTERPRETATION LOCK",
        "-" * 118,
        "Independent severity unit: domain (n=18), not plant (n=216).",
        "The 12 plants/domain share environmental-history predictors.",
        "Pearson significance alone is not sufficient if Spearman and leave-one-domain diagnostics show instability.",
        "Real-data results are external association/validation evidence, not controlled causal proof.",
        "P7.9 inner-CV ID vs OOD ΔMAE must not be used as a primary degradation claim.",
        "",
        "VALIDATION ERRORS",
        "-" * 118,
        f"Error count: {len(errors)}",
    ])
    lines.extend([f"ERROR: {e}" for e in errors] or ["None"])

    lines.extend([
        "",
        "FINAL VERDICT",
        "=" * 118,
        verdict,
        "",
        "NEXT IF PASS",
        "-" * 118,
        "Use this corrected matched analysis for the Phase-7 synthesis and paper wording.",
        "Do not resurrect the unmatched P7.9 ID-vs-OOD subtraction as evidence for H7.1.",
    ])

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "phase": "P7.10",
        "purpose": "correct unmatched P7.9 ID/OOD subtraction and diagnose severity robustness",
        "independent_environmental_units": 18,
        "matched_rows": len(matched),
        "features": features,
        "severity_reconstruction_max_abs_diff": max_v_diff,
        "h71": "not directly identifiable as matched ID-vs-OOD from available real-data protocol",
        "h72": "domain-level Pearson/Spearman plus leave-one-domain influence diagnostics",
        "h73": "exploratory OLS model-by-severity interaction with HC3 SE",
        "errors": errors,
        "verdict": verdict,
        "raw_archive_modified": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(REPORT.read_text(encoding="utf-8"))
    print(f"Matched results: {DOMAIN_CSV}")
    print(f"Influence diagnostics: {INFLUENCE_CSV}")
    print(f"OLS coefficients: {OLS_CSV}")
    print(f"Report: {REPORT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
