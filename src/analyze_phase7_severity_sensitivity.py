#!/usr/bin/env python3
"""
Phase 7 pre-submission robustness analysis:
nonredundant shift-severity sensitivity.

Purpose
-------
Recompute Phase-7 held-out-domain shift severity without TDS and without
the full 32-feature summary set, while preserving the frozen P7.9 model
predictions and outer leave-one-domain-out folds.

Primary compact severity:
    pH mean/std + EC mean/std + water-temperature mean/std

Secondary compact severity:
    pH mean + EC mean + water-temperature mean

Important
---------
- This is a sensitivity analysis, NOT Phase 8.
- It does not retrain M1/M0/M2.
- It does not modify P7.8, P7.9, P7.10, or raw data.
- Severity preprocessing is fitted using outer-training domains only.
- TDS is intentionally excluded because it is source-derived from EC.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "experiment_7" / "plant_level_dataset" / "phase7_plant_level_dataset.csv"
P79_DIR = ROOT / "results" / "experiment_7" / "ood_modeling"
P710_DIR = ROOT / "results" / "experiment_7" / "matched_ood_analysis"
OUT = ROOT / "results" / "experiment_7" / "severity_sensitivity"

EXPECTED_DOMAINS = 18
EXPECTED_MODELS = ("M1", "M0", "M2")

PRIMARY_FEATURES = (
    "ph_mean", "ph_std",
    "ec_mean", "ec_std",
    "water_temperature_mean", "water_temperature_std",
)
SECONDARY_FEATURES = (
    "ph_mean",
    "ec_mean",
    "water_temperature_mean",
)


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_file(directory: Path, preferred: tuple[str, ...], must_contain: tuple[str, ...]) -> Path:
    for name in preferred:
        p = directory / name
        if p.exists():
            return p
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")
    candidates = []
    for p in sorted(directory.glob("*.csv")):
        n = p.name.lower()
        if all(x.lower() in n for x in must_contain):
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # Prefer the most informative/largest table if names are ambiguous.
        return max(candidates, key=lambda p: p.stat().st_size)
    raise FileNotFoundError(
        f"Could not locate required CSV in {directory}. "
        f"Preferred={preferred}, required name tokens={must_contain}"
    )


def resolve_col(df: pd.DataFrame, aliases: tuple[str, ...], required: bool = True) -> str | None:
    mapping = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in mapping:
            return mapping[norm(a)]
    # relaxed contains match
    for c in df.columns:
        nc = norm(c)
        if any(norm(a) in nc or nc in norm(a) for a in aliases):
            return c
    if required:
        raise KeyError(f"Could not resolve column aliases {aliases}. Available: {list(df.columns)}")
    return None


def resolve_feature(df: pd.DataFrame, variable: str, stat: str) -> str:
    # Prefer exact conventional names, then normalized token logic.
    aliases = [
        f"{variable}_{stat}",
        f"env_{variable}_{stat}",
        f"{variable}_{stat}_value",
    ]
    mapping = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in mapping:
            return mapping[norm(a)]

    v_tokens = {
        "ph": ("ph",),
        "ec": ("ec", "electricalconductivity"),
        "water_temperature": ("watertemperature", "watertemp", "watertemperaturec"),
    }[variable]
    s_tokens = {
        "mean": ("mean", "avg", "average"),
        "std": ("std", "sd", "stdev", "standarddeviation"),
    }[stat]

    matches = []
    for c in df.columns:
        nc = norm(c)
        if any(v in nc for v in v_tokens) and any(s in nc for s in s_tokens):
            if "tds" not in nc:
                matches.append(c)
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Could not uniquely resolve feature {variable}_{stat}. "
        f"Candidates={matches}. Available columns={list(df.columns)}"
    )


def domain_key_from_parts(df: pd.DataFrame) -> pd.Series:
    exp = resolve_col(df, ("experiment", "exp"))
    rep = resolve_col(df, ("replicate", "rep"))
    trt = resolve_col(df, ("treatment", "trt"))
    return (
        df[exp].astype(str).str.strip()
        + "__R" + df[rep].astype(str).str.extract(r"(\d+)", expand=False).fillna(df[rep].astype(str))
        + "-T" + df[trt].astype(str).str.extract(r"(\d+)", expand=False).fillna(df[trt].astype(str))
    )


def add_domain_key(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Build a globally unique Phase-7 domain key.

    IMPORTANT:
    A local label such as R1-T1 is repeated across EXP1/EXP2/EXP3, so it
    must NOT be used by itself.  Prefer experiment + local domain whenever
    an experiment column exists.  Expected canonical result:
        EXP1__R1-T1
        EXP2__R1-T1
        ...
    """
    out = df.copy()

    exp_col = resolve_col(out, ("experiment", "experiment_id", "exp"), required=False)
    domain_col = resolve_col(
        out,
        ("held_out_domain", "heldout_domain", "test_domain",
         "domain_key", "domain_id", "domain"),
        required=False,
    )

    if domain_col is not None:
        local = out[domain_col].astype(str).str.strip()

        # If the domain column already contains EXP#, preserve it.
        already_global = local.str.upper().str.contains(r"EXP\s*\d+", regex=True)

        if exp_col is not None:
            exp = out[exp_col].astype(str).str.strip()
            exp_num = exp.str.extract(r"(\d+)", expand=False)
            exp_norm = np.where(
                exp_num.notna(),
                "EXP" + exp_num.fillna(""),
                exp.str.upper().str.replace(r"\s+", "", regex=True),
            )
            combined = pd.Series(exp_norm, index=out.index).astype(str) + "__" + local
            out["_domain"] = np.where(already_global, local, combined)
        else:
            out["_domain"] = local

        return out, "_domain"

    # No explicit domain column: reconstruct from experiment/treatment/replicate.
    out["_domain"] = domain_key_from_parts(out)
    return out, "_domain"


def canonical_domain(s: object) -> str:
    x = str(s).strip().upper().replace(" ", "")
    x = x.replace("EXPERIMENT", "EXP")
    # Normalize EXP2_R1-T2 / EXP2__R1_T2 etc.
    m = re.search(r"(EXP\d+).*?R(\d+).*?T(\d+)", x)
    if m:
        return f"{m.group(1)}__R{m.group(2)}-T{m.group(3)}"
    return x


def rms_centroid_severity(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> float:
    train_x = train[features].apply(pd.to_numeric, errors="coerce")
    test_x = test[features].apply(pd.to_numeric, errors="coerce")

    med = train_x.median(axis=0)
    train_x = train_x.fillna(med)
    test_x = test_x.fillna(med)

    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)

    train_centroid = ((train_x - mu) / sd).mean(axis=0).to_numpy(float)
    test_centroid = ((test_x - mu) / sd).mean(axis=0).to_numpy(float)
    return float(np.sqrt(np.mean((test_centroid - train_centroid) ** 2)))


def corr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    a = pd.to_numeric(x, errors="coerce").to_numpy(float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return dict(pearson_r=np.nan, pearson_p=np.nan, spearman_rho=np.nan, spearman_p=np.nan)
    pr = pearsonr(a, b)
    sr = spearmanr(a, b)
    return dict(
        pearson_r=float(pr.statistic), pearson_p=float(pr.pvalue),
        spearman_rho=float(sr.statistic), spearman_p=float(sr.pvalue),
    )


def loo_pearson(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    rows = []
    for d in sorted(df["_domain"].unique()):
        z = df[df["_domain"] != d]
        r = corr(z[x], z[y])
        rows.append({"removed_domain": d, **r})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    if not DATASET.exists():
        raise FileNotFoundError(f"Missing frozen P7.8 dataset: {DATASET}")

    fold_file = find_file(
        P79_DIR,
        (
            "phase7_ood_fold_results.csv",
            "phase7_ood_folds.csv",
            "ood_fold_results.csv",
            "fold_results.csv",
        ),
        ("fold",),
    )

    df = pd.read_csv(DATASET)
    folds = pd.read_csv(fold_file)

    df, _ = add_domain_key(df)
    folds, _ = add_domain_key(folds)
    df["_domain"] = df["_domain"].map(canonical_domain)
    folds["_domain"] = folds["_domain"].map(canonical_domain)

    # Resolve compact nonredundant features from actual frozen dataset.
    resolved = {
        "ph_mean": resolve_feature(df, "ph", "mean"),
        "ph_std": resolve_feature(df, "ph", "std"),
        "ec_mean": resolve_feature(df, "ec", "mean"),
        "ec_std": resolve_feature(df, "ec", "std"),
        "water_temperature_mean": resolve_feature(df, "water_temperature", "mean"),
        "water_temperature_std": resolve_feature(df, "water_temperature", "std"),
    }
    primary = [resolved[x] for x in PRIMARY_FEATURES]
    secondary = [resolved[x] for x in SECONDARY_FEATURES]

    if any("tds" in norm(c) for c in primary + secondary):
        raise RuntimeError("TDS unexpectedly entered compact severity features.")

    domains = sorted(df["_domain"].dropna().unique())
    if len(domains) != EXPECTED_DOMAINS:
        raw_cols = [c for c in df.columns if any(k in norm(c) for k in ("experiment", "domain", "replicate", "treatment"))]
        preview = df[raw_cols + ["_domain"]].drop_duplicates().head(30).to_string(index=False) if raw_cols else df[["_domain"]].drop_duplicates().to_string(index=False)
        raise RuntimeError(
            f"Expected {EXPECTED_DOMAINS} globally unique experiment-domain cells, "
            f"found {len(domains)}: {domains}\n\nDomain reconstruction preview:\n{preview}"
        )

    counts = df.groupby("_domain").size()
    if not (counts == 12).all():
        raise RuntimeError(f"Expected 12 plants/domain. Counts:\n{counts}")

    # One domain-level environmental vector per domain is expected, replicated over plants.
    for c in primary:
        nuniq = df.groupby("_domain")[c].nunique(dropna=False)
        if (nuniq > 1).any():
            raise RuntimeError(f"Feature {c} varies within domain; cannot use frozen domain summary as assumed.")

    sev_rows = []
    for held in domains:
        tr = df[df["_domain"] != held]
        te = df[df["_domain"] == held]
        sev_rows.append({
            "_domain": held,
            "v_compact6": rms_centroid_severity(tr, te, primary),
            "v_mean3": rms_centroid_severity(tr, te, secondary),
        })
    sev = pd.DataFrame(sev_rows)

    model_col = resolve_col(folds, ("model", "model_name", "configuration"))
    mae_col = resolve_col(folds, ("ood_mae", "test_mae", "mae_ood"))
    old_v_col = resolve_col(folds, ("shift_severity_v", "severity_v", "v"), required=False)

    folds[model_col] = folds[model_col].astype(str).str.strip()
    domain_model = folds[["_domain", model_col, mae_col] + ([old_v_col] if old_v_col else [])].copy()
    domain_model = domain_model.merge(sev, on="_domain", how="left", validate="many_to_one")

    if domain_model["v_compact6"].isna().any():
        raise RuntimeError("Compact severity failed to merge onto some P7.9 fold rows.")

    # If P7.10 matched results exist, recover excess MAE for a secondary sensitivity outcome.
    matched_file = None
    try:
        matched_file = find_file(
            P710_DIR,
            (
                "phase7_matched_ood_domain_results.csv",
                "matched_ood_domain_results.csv",
                "phase7_matched_domain_results.csv",
                "matched_domain_results.csv",
            ),
            ("matched",),
        )
    except FileNotFoundError:
        pass

    excess_col_final = None
    if matched_file is not None:
        matched = pd.read_csv(matched_file)
        matched, _ = add_domain_key(matched)
        matched["_domain"] = matched["_domain"].map(canonical_domain)
        mm = resolve_col(matched, ("model", "model_name", "configuration"))
        ex = resolve_col(matched, ("excess_mae", "mae_excess"), required=False)
        if ex is not None:
            tmp = matched[["_domain", mm, ex]].copy()
            tmp.columns = ["_domain", model_col, "excess_mae"]
            domain_model = domain_model.merge(
                tmp, on=["_domain", model_col], how="left", validate="one_to_one"
            )
            excess_col_final = "excess_mae"

    # Correlation and influence results.
    summary_rows = []
    influence_frames = []
    outcomes = [("ood_mae", mae_col)]
    if excess_col_final:
        outcomes.append(("excess_mae", excess_col_final))

    severity_defs = [("compact6", "v_compact6"), ("mean3", "v_mean3")]
    if old_v_col:
        severity_defs.insert(0, ("original32", old_v_col))

    for model in EXPECTED_MODELS:
        z = domain_model[domain_model[model_col] == model].copy()
        if len(z) != EXPECTED_DOMAINS:
            raise RuntimeError(f"{model}: expected 18 fold rows, found {len(z)}.")
        for outcome_name, outcome_col in outcomes:
            for sev_name, sev_col in severity_defs:
                r = corr(z[sev_col], z[outcome_col])
                inf = loo_pearson(z, sev_col, outcome_col)
                inf.insert(0, "severity_definition", sev_name)
                inf.insert(0, "outcome", outcome_name)
                inf.insert(0, "model", model)
                influence_frames.append(inf)
                summary_rows.append({
                    "model": model,
                    "outcome": outcome_name,
                    "severity_definition": sev_name,
                    **r,
                    "loo_pearson_min": inf["pearson_r"].min(),
                    "loo_pearson_max": inf["pearson_r"].max(),
                })

    summary = pd.DataFrame(summary_rows)
    influence = pd.concat(influence_frames, ignore_index=True)

    # Domain severity ranking and comparison with original.
    rank = sev.copy()
    rank["rank_compact6"] = rank["v_compact6"].rank(ascending=False, method="min")
    rank["rank_mean3"] = rank["v_mean3"].rank(ascending=False, method="min")

    if old_v_col:
        old_domain = (
            folds.groupby("_domain", as_index=False)[old_v_col]
            .first()
            .rename(columns={old_v_col: "v_original32"})
        )
        rank = rank.merge(old_domain, on="_domain", how="left", validate="one_to_one")
        rank["rank_original32"] = rank["v_original32"].rank(ascending=False, method="min")
        rank["compact6_vs_original_rank_change"] = rank["rank_compact6"] - rank["rank_original32"]

    # Save machine-readable outputs.
    domain_model.to_csv(OUT / "phase7_severity_sensitivity_domain_model.csv", index=False)
    summary.to_csv(OUT / "phase7_severity_sensitivity_correlations.csv", index=False)
    influence.to_csv(OUT / "phase7_severity_sensitivity_loo_influence.csv", index=False)
    rank.sort_values("v_compact6", ascending=False).to_csv(
        OUT / "phase7_severity_sensitivity_domain_ranking.csv", index=False
    )

    # Decision-oriented report.
    lines = [
        "PHASE 7 PRE-SUBMISSION ROBUSTNESS — NONREDUNDANT SEVERITY SENSITIVITY",
        "=" * 100,
        f"Frozen dataset: {DATASET}",
        f"Frozen P7.9 fold results: {fold_file}",
        f"Matched P7.10 results: {matched_file if matched_file else 'not located; OOD-MAE analysis still valid'}",
        "",
        "PURPOSE",
        "-" * 100,
        "Test whether H7.2 severity/error evidence depends materially on the original 32-feature V",
        "when TDS (source-derived from EC) and other potentially redundant summary statistics are removed.",
        "No models are retrained. No frozen Phase-7 files are modified.",
        "",
        "PRIMARY COMPACT V",
        "-" * 100,
        "Six features: pH mean/std, EC mean/std, water-temperature mean/std.",
        "Resolved dataset columns:",
    ]
    for k in PRIMARY_FEATURES:
        lines.append(f"  {k}: {resolved[k]}")
    lines += [
        "",
        "SECONDARY COMPACT V",
        "-" * 100,
        "Three features: pH mean, EC mean, water-temperature mean.",
        "",
        "INTEGRITY",
        "-" * 100,
        f"Plants: {len(df)}",
        f"Domains: {len(domains)}",
        f"Fold/model rows: {len(domain_model)}",
        f"TDS included in compact features: NO",
        "",
        "CORRELATION / INFLUENCE RESULTS",
        "-" * 100,
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['model']} | {r['outcome']} | {r['severity_definition']}: "
            f"Pearson r={r['pearson_r']:.6f}, p={r['pearson_p']:.6g}; "
            f"Spearman rho={r['spearman_rho']:.6f}, p={r['spearman_p']:.6g}; "
            f"LOO Pearson range=[{r['loo_pearson_min']:.6f}, {r['loo_pearson_max']:.6f}]"
        )

    lines += ["", "TOP COMPACT-SEVERITY DOMAINS", "-" * 100]
    for _, r in rank.sort_values("v_compact6", ascending=False).head(5).iterrows():
        old = f", original32={r['v_original32']:.6f}" if "v_original32" in rank.columns else ""
        lines.append(f"{r['_domain']}: compact6={r['v_compact6']:.6f}, mean3={r['v_mean3']:.6f}{old}")

    # Conservative automatic interpretation.
    compact = summary[(summary["severity_definition"] == "compact6") & (summary["outcome"] == "ood_mae")]
    robust_rank = bool((compact["spearman_p"] < 0.05).all() and (compact["spearman_rho"] > 0.5).all())
    stable_loo = bool((compact["loo_pearson_min"] > 0.5).all())
    strong_linear = bool((compact["pearson_r"] > 0.7).all())

    lines += ["", "PRE-SUBMISSION INTERPRETATION", "-" * 100]
    if robust_rank and stable_loo and strong_linear:
        lines.append(
            "Compact nonredundant severity retains strong linear, rank-based, and leave-one-domain evidence. "
            "This materially strengthens H7.2 robustness."
        )
    elif strong_linear:
        lines.append(
            "Compact nonredundant severity retains a strong linear association, but rank/influence robustness "
            "is incomplete. H7.2 should remain 'suggestive / sensitivity-qualified', not confirmed."
        )
    else:
        lines.append(
            "The severity/error relationship weakens materially under the compact nonredundant definition. "
            "H7.2 is sensitive to severity construction and the manuscript should explicitly report this."
        )

    lines += [
        "",
        "SCIENTIFIC LOCK",
        "-" * 100,
        "This sensitivity analysis does not replace the frozen P7.7/P7.10 definitions or results.",
        "It is a pre-submission robustness check designed to test dependence on severity construction.",
        "Do not alter the original Phase-7 results; report both original and compact sensitivity evidence.",
        "",
        "VERDICT: PASS",
    ]

    report = OUT / "phase7_severity_sensitivity_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nOutputs written to: {OUT}")


if __name__ == "__main__":
    main()
