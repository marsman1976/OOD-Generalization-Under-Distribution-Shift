from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("default")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "experiment_2"
    / "confirmation"
    / "analysis"
    / "paired_results.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "experiment_2"
    / "confirmation"
    / "analysis"
    / "interactions"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def pick_severity_column(frame):
    """Find the numeric severity column used by EXP1C."""
    candidates = [
        "severity",
        "severity_level",
        "shift_severity",
        "severity_value",
    ]
    for col in candidates:
        if col in frame.columns:
            return col
    raise KeyError(
        "No severity column found. Expected one of: "
        + ", ".join(candidates)
        + f"\nAvailable columns: {list(frame.columns)}"
    )


def joint_wald(model, term_names, label):
    """
    Joint Wald test for a selected set of coefficient names.
    Returns a record instead of crashing if the covariance is singular.
    """
    names = list(model.params.index)
    selected = [n for n in names if term_names(n)]

    if not selected:
        return {
            "test": label,
            "n_terms_requested": 0,
            "statistic": np.nan,
            "df_constraint": np.nan,
            "p_value": np.nan,
            "status": "NO_MATCHING_TERMS",
        }

    R = np.zeros((len(selected), len(names)))
    for i, name in enumerate(selected):
        R[i, names.index(name)] = 1.0

    try:
        wt = model.wald_test(R, scalar=True)
        stat = float(np.asarray(wt.statistic).squeeze())
        pval = float(np.asarray(wt.pvalue).squeeze())
        dfc = getattr(wt, "df_denom", len(selected))
        try:
            dfc = float(dfc)
        except Exception:
            dfc = float(len(selected))

        return {
            "test": label,
            "n_terms_requested": len(selected),
            "statistic": stat,
            "df_constraint": dfc,
            "p_value": pval,
            "status": "OK",
        }
    except Exception as exc:
        return {
            "test": label,
            "n_terms_requested": len(selected),
            "statistic": np.nan,
            "df_constraint": np.nan,
            "p_value": np.nan,
            "status": f"FAILED: {type(exc).__name__}: {exc}",
        }


def is_DA(name):
    return (
        "C(dataset_id)" in name
        and "C(augmentation_id)" in name
        and "severity_z" not in name
    )


def is_AV(name):
    return (
        "C(augmentation_id)" in name
        and "severity_z" in name
        and "C(dataset_id)" not in name
    )


def is_DV(name):
    return (
        "C(dataset_id)" in name
        and "severity_z" in name
        and "C(augmentation_id)" not in name
    )


def is_DAV(name):
    return (
        "C(dataset_id)" in name
        and "C(augmentation_id)" in name
        and "severity_z" in name
    )


def is_DA_categorical(name):
    return (
        "C(dataset_id)" in name
        and "C(augmentation_id)" in name
        and "C(severity_cat)" not in name
    )


def is_AV_categorical(name):
    return (
        "C(augmentation_id)" in name
        and "C(severity_cat)" in name
        and "C(dataset_id)" not in name
    )


def is_DAV_categorical(name):
    return (
        "C(dataset_id)" in name
        and "C(augmentation_id)" in name
        and "C(severity_cat)" in name
    )


# ============================================================
# LOAD DATA
# ============================================================

banner("PHASE 2 FORMAL INTERACTION ANALYSIS")

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

print("\nInput:", INPUT)
print("Rows:", len(df))

# ============================================================
# KEEP EXP1C ONLY
# ============================================================

required = {
    "experiment",
    "dataset_id",
    "augmentation_id",
    "seed",
    "shift_type",
    "delta_mse",
}
missing = required.difference(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

df = df[df["experiment"] == "EXP1C"].copy()

print("EXP1C paired rows:", len(df))
print("Datasets:", sorted(df["dataset_id"].unique()))
print("Augmentations:", sorted(df["augmentation_id"].unique()))
print("Seeds:", sorted(df["seed"].unique()))
print("Shift types:", sorted(df["shift_type"].unique()))

# ============================================================
# INTEGRITY
# ============================================================

assert len(df) == 5200, f"Expected 5200 EXP1C paired rows, found {len(df)}"
assert df["dataset_id"].nunique() == 10
assert df["augmentation_id"].nunique() == 4
assert df["seed"].nunique() == 10
assert set(df["augmentation_id"]) == {"A1", "A2", "A3", "A4"}
assert set(df["shift_type"]) == {"temperature", "humidity"}
assert df["delta_mse"].notna().all()

severity_col = pick_severity_column(df)
df[severity_col] = pd.to_numeric(df[severity_col], errors="raise")

expected_rows = {"temperature": 2400, "humidity": 2800}
for shift_type, n_expected in expected_rows.items():
    n_actual = int((df["shift_type"] == shift_type).sum())
    assert n_actual == n_expected, (
        f"{shift_type}: expected {n_expected} rows, found {n_actual}"
    )

print("\nINTERACTION INPUT INTEGRITY: PASS")
print("Severity column:", severity_col)

# ============================================================
# PREPARE DATA
# ============================================================

df["dataset_id"] = pd.Categorical(
    df["dataset_id"],
    categories=[f"D{i:02d}" for i in range(10)],
    ordered=True,
)
df["augmentation_id"] = pd.Categorical(
    df["augmentation_id"],
    categories=["A1", "A2", "A3", "A4"],
    ordered=True,
)

# D x seed is the repeated-evaluation cluster:
# each fitted D/seed baseline is reused across A and severity evaluations.
df["cluster_id"] = (
    df["dataset_id"].astype(str)
    + "_seed"
    + df["seed"].astype(str)
)

# Standardize severity separately within each shift family.
df["severity_z"] = np.nan
for shift_type in ["temperature", "humidity"]:
    mask = df["shift_type"] == shift_type
    x = df.loc[mask, severity_col].astype(float)
    sd = float(x.std(ddof=0))
    if sd == 0:
        raise ValueError(f"{shift_type}: severity has zero variance.")
    df.loc[mask, "severity_z"] = (x - float(x.mean())) / sd

assert df["severity_z"].notna().all()

# Save exact interaction-analysis input for reproducibility.
df.to_csv(OUTPUT / "interaction_analysis_input.csv", index=False)

# ============================================================
# PRIMARY LINEAR-SEVERITY MODELS
# ============================================================

FORMULA = (
    "delta_mse ~ "
    "C(dataset_id) * C(augmentation_id) * severity_z"
)

omnibus_rows = []
direction_rows = []

for shift_type in ["temperature", "humidity"]:
    banner(f"{shift_type.upper()} SHIFT — PRIMARY MODEL")

    sub = df[df["shift_type"] == shift_type].copy()

    print("Observations:", len(sub))
    print("Clusters:", sub["cluster_id"].nunique())
    print("Severity values:", sorted(sub[severity_col].unique()))
    print("Fitting:", FORMULA)

    model = smf.ols(FORMULA, data=sub).fit(
        cov_type="cluster",
        cov_kwds={"groups": sub["cluster_id"]},
    )

    print("Model fitted.")
    print("R-squared:", round(float(model.rsquared), 6))

    # Full model text and coefficient table.
    (OUTPUT / f"{shift_type}_model_summary.txt").write_text(
        model.summary().as_text(),
        encoding="utf-8",
    )

    coef = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
            "std_error": model.bse.values,
            "z_or_t": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )
    coef.to_csv(
        OUTPUT / f"{shift_type}_coefficients.csv",
        index=False,
    )

    tests = [
        ("D_x_A_at_centered_severity", is_DA),
        ("A_x_V_at_reference_D00", is_AV),
        ("D_x_V_at_reference_A1", is_DV),
        ("D_x_A_x_V_linear", is_DAV),
    ]

    for label, selector in tests:
        rec = joint_wald(model, selector, label)
        rec["shift_type"] = shift_type
        rec["n_observations"] = len(sub)
        rec["n_clusters"] = sub["cluster_id"].nunique()
        omnibus_rows.append(rec)

    # Descriptive cell-wise linear severity slopes.
    for (d, a), g in sub.groupby(
        ["dataset_id", "augmentation_id"],
        observed=True,
    ):
        slope_model = smf.ols(
            "delta_mse ~ severity_z",
            data=g,
        ).fit()

        direction_rows.append(
            {
                "shift_type": shift_type,
                "dataset_id": str(d),
                "augmentation_id": str(a),
                "n": len(g),
                "mean_delta_mse": float(g["delta_mse"].mean()),
                "linear_severity_slope": float(
                    slope_model.params["severity_z"]
                ),
            }
        )

omnibus = pd.DataFrame(omnibus_rows)
omnibus.to_csv(
    OUTPUT / "omnibus_interaction_tests.csv",
    index=False,
)

direction = pd.DataFrame(direction_rows)
direction.to_csv(
    OUTPUT / "interaction_direction_summary.csv",
    index=False,
)

print("\nPrimary interaction tests written.")

# ============================================================
# CATEGORICAL-SEVERITY SENSITIVITY
# ============================================================

# This is intentionally treated as a sensitivity analysis.
# It is much higher-dimensional than the primary linear-severity model.
# Rank-deficient Wald tests are recorded as FAILED rather than terminating
# the script.

sensitivity_rows = []

for shift_type in ["temperature", "humidity"]:
    banner(f"{shift_type.upper()} SHIFT — CATEGORICAL-SEVERITY SENSITIVITY")

    sub = df[df["shift_type"] == shift_type].copy()
    sub["severity_cat"] = sub[severity_col].astype(str)

    cat_formula = (
        "delta_mse ~ "
        "C(dataset_id) * C(augmentation_id) * C(severity_cat)"
    )

    print("Observations:", len(sub))
    print("Clusters:", sub["cluster_id"].nunique())
    print("Fitting high-dimensional sensitivity model...")

    try:
        cat_model = smf.ols(cat_formula, data=sub).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["cluster_id"]},
        )

        (OUTPUT / f"{shift_type}_categorical_model_summary.txt").write_text(
            cat_model.summary().as_text(),
            encoding="utf-8",
        )

        tests = [
            ("D_x_A_categorical_at_reference_severity", is_DA_categorical),
            ("A_x_V_categorical_at_reference_D00", is_AV_categorical),
            ("D_x_A_x_V_categorical", is_DAV_categorical),
        ]

        for label, selector in tests:
            rec = joint_wald(cat_model, selector, label)
            rec["shift_type"] = shift_type
            rec["n_observations"] = len(sub)
            rec["n_clusters"] = sub["cluster_id"].nunique()
            rec["note"] = (
                "Sensitivity analysis; high-dimensional clustered Wald "
                "tests may be rank-limited."
            )
            sensitivity_rows.append(rec)

        print("Categorical sensitivity model completed.")

    except Exception as exc:
        print(
            "Categorical sensitivity model failed safely:",
            type(exc).__name__,
            str(exc),
        )
        sensitivity_rows.append(
            {
                "test": "categorical_model",
                "n_terms_requested": np.nan,
                "statistic": np.nan,
                "df_constraint": np.nan,
                "p_value": np.nan,
                "status": f"FAILED: {type(exc).__name__}: {exc}",
                "shift_type": shift_type,
                "n_observations": len(sub),
                "n_clusters": sub["cluster_id"].nunique(),
                "note": (
                    "Sensitivity analysis failed; primary linear-severity "
                    "analysis remains available."
                ),
            }
        )

sensitivity = pd.DataFrame(sensitivity_rows)
sensitivity.to_csv(
    OUTPUT / "categorical_severity_sensitivity.csv",
    index=False,
)

# ============================================================
# PER-SEVERITY DESCRIPTIVE / LOWER-DIMENSIONAL CHECK
# ============================================================

# This table is useful even if the high-dimensional categorical Wald
# covariance is rank-limited. It reports D x A interaction tests at each
# observed severity separately.

per_severity_rows = []

for shift_type in ["temperature", "humidity"]:
    sub_shift = df[df["shift_type"] == shift_type].copy()

    for sev in sorted(sub_shift[severity_col].unique()):
        sub = sub_shift[sub_shift[severity_col] == sev].copy()

        model = smf.ols(
            "delta_mse ~ C(dataset_id) * C(augmentation_id)",
            data=sub,
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": sub["cluster_id"]},
        )

        rec = joint_wald(
            model,
            lambda n: (
                "C(dataset_id)" in n
                and "C(augmentation_id)" in n
            ),
            "D_x_A_at_fixed_severity",
        )

        rec["shift_type"] = shift_type
        rec["severity"] = sev
        rec["n_observations"] = len(sub)
        rec["n_clusters"] = sub["cluster_id"].nunique()
        per_severity_rows.append(rec)

pd.DataFrame(per_severity_rows).to_csv(
    OUTPUT / "per_severity_DxA_tests.csv",
    index=False,
)

# ============================================================
# REPORT
# ============================================================

report_lines = [
    "PHASE 2 FORMAL INTERACTION ANALYSIS",
    "=" * 75,
    "",
    f"Input: {INPUT}",
    f"EXP1C paired rows: {len(df)}",
    f"Datasets: {df['dataset_id'].nunique()}",
    f"Augmentations compared with A0: {df['augmentation_id'].nunique()}",
    f"Seeds: {df['seed'].nunique()}",
    f"Clusters (D x seed): {df['cluster_id'].nunique()}",
    f"Severity column: {severity_col}",
    "",
    "PRIMARY MODEL",
    FORMULA,
    "",
    "Interpretation:",
    "- delta_mse < 0 means augmentation improved MSE relative to A0.",
    "- delta_mse > 0 means augmentation harmed MSE relative to A0.",
    "- D x A is evaluated at centered severity (severity_z = 0).",
    "- A x V terms use D00 as the treatment-coded reference dataset.",
    "- D x V terms use A1 as the treatment-coded reference augmentation.",
    "- D x A x V is the main test of whether augmentation-effect severity "
    "slopes differ across training compositions.",
    "",
    "PRIMARY OMNIBUS TESTS",
    omnibus.to_string(index=False),
    "",
    "CATEGORICAL-SEVERITY SENSITIVITY",
    sensitivity.to_string(index=False),
    "",
    "CAUTION",
    "The categorical-severity model is high-dimensional relative to the "
    "number of D x seed clusters. Its clustered Wald covariance can be "
    "rank-limited. Treat it as sensitivity evidence, not as the sole "
    "confirmatory result.",
    "",
    "FILES WRITTEN",
    "- interaction_analysis_input.csv",
    "- omnibus_interaction_tests.csv",
    "- categorical_severity_sensitivity.csv",
    "- per_severity_DxA_tests.csv",
    "- interaction_direction_summary.csv",
    "- temperature_coefficients.csv",
    "- humidity_coefficients.csv",
    "- temperature_model_summary.txt",
    "- humidity_model_summary.txt",
    "- temperature_categorical_model_summary.txt (if model succeeds)",
    "- humidity_categorical_model_summary.txt (if model succeeds)",
]

(OUTPUT / "phase2_interaction_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)

banner("INTERACTION ANALYSIS COMPLETE")
print("Output directory:", OUTPUT)
for p in sorted(OUTPUT.iterdir()):
    if p.is_file():
        print(" -", p.name)