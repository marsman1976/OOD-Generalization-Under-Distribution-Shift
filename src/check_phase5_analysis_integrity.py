from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# PHASE 5 ANALYSIS INTEGRITY CHECK
#
# Checks outputs created by:
# src/analyze_phase5_confirmation.py
#
# This script DOES NOT perform new hypothesis testing.
# It verifies:
#
# 1. Raw confirmation structure
# 2. AUC trajectory construction
# 3. A0 pairing
# 4. Delta-AUC calculations
# 5. Pointwise delta calculations
# 6. H5.5 output structure
# 7. H5.6 output structure
# 8. Per-severity output structure
# 9. No missing / infinite statistics
# 10. Internal consistency of key calculations
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

CONFIRMATION_DIR = (
    ROOT
    / "results"
    / "experiment_5"
    / "confirmation"
)

ANALYSIS_DIR = (
    CONFIRMATION_DIR
    / "analysis"
)


RAW_FILE = (
    CONFIRMATION_DIR
    / "master_results.csv"
)


# ============================================================
# Expected design
# ============================================================

DATASETS = [
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
]

AUGMENTATIONS = [
    "A0", "A1", "A2", "A3", "A4",
]

NON_REFERENCE_AUGMENTATIONS = [
    "A1", "A2", "A3", "A4",
]

MODELS = [
    "M0", "M1", "M2",
]

SEEDS = [
    121, 132, 143, 154, 165,
    176, 187, 198, 209, 220,
]

SHIFT_TYPES = [
    "temperature",
    "humidity",
]


# ============================================================
# Expected files
# ============================================================

FILES = {

    "auc":
        ANALYSIS_DIR
        / "phase5_auc.csv",

    "delta_auc":
        ANALYSIS_DIR
        / "phase5_delta_auc.csv",

    "delta_auc_summary":
        ANALYSIS_DIR
        / "phase5_delta_auc_summary.csv",

    "delta_auc_cells":
        ANALYSIS_DIR
        / "phase5_delta_auc_cells.csv",

    "h55":
        ANALYSIS_DIR
        / "phase5_h55_primary.csv",

    "h55_perm_summary":
        ANALYSIS_DIR
        / "phase5_h55_permutation_summary.csv",

    "pointwise":
        ANALYSIS_DIR
        / "phase5_pointwise_delta.csv",

    "h56":
        ANALYSIS_DIR
        / "phase5_h56_secondary.csv",

    "per_severity":
        ANALYSIS_DIR
        / "phase5_per_severity_DxAxM.csv",

    "severity_summary":
        ANALYSIS_DIR
        / "phase5_severity_summary.csv",
}


# ============================================================
# Helpers
# ============================================================

checks_passed = 0


def section(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def passed(message):

    global checks_passed

    checks_passed += 1

    print(
        f"[PASS] {message}"
    )


def fail(message):

    print()
    print(
        f"[FAIL] {message}"
    )

    print()
    print(
        "PHASE 5 ANALYSIS INTEGRITY: FAIL"
    )

    sys.exit(1)


def require(
    condition,
    message,
):

    if condition:
        passed(message)

    else:
        fail(message)


def finite(
    series,
):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return (
        values.notna().all()
        and np.isfinite(
            values.to_numpy(
                dtype=float
            )
        ).all()
    )


def close_enough(
    a,
    b,
    atol=1e-10,
    rtol=1e-8,
):

    return np.allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


# ============================================================
# File existence
# ============================================================

section(
    "1. ANALYSIS FILE EXISTENCE"
)


require(
    RAW_FILE.exists(),
    "master_results.csv exists",
)


for name, path in FILES.items():

    require(
        path.exists(),
        f"{path.name} exists",
    )


# ============================================================
# Load
# ============================================================

raw = pd.read_csv(
    RAW_FILE
)

auc = pd.read_csv(
    FILES["auc"]
)

delta_auc = pd.read_csv(
    FILES["delta_auc"]
)

delta_summary = pd.read_csv(
    FILES["delta_auc_summary"]
)

delta_cells = pd.read_csv(
    FILES["delta_auc_cells"]
)

h55 = pd.read_csv(
    FILES["h55"]
)

h55_perm = pd.read_csv(
    FILES["h55_perm_summary"]
)

pointwise = pd.read_csv(
    FILES["pointwise"]
)

h56 = pd.read_csv(
    FILES["h56"]
)

per_severity = pd.read_csv(
    FILES["per_severity"]
)

severity_summary = pd.read_csv(
    FILES["severity_summary"]
)


# ============================================================
# Raw confirmation
# ============================================================

section(
    "2. RAW CONFIRMATION STRUCTURE"
)


require(
    len(raw) == 30000,
    "Raw confirmation contains exactly 30,000 rows",
)


require(
    raw["run_id"].nunique() == 1500,
    "Raw confirmation contains exactly 1,500 models",
)


model_counts = (
    raw.groupby(
        "run_id"
    )
    .size()
)


require(
    (model_counts == 20).all(),
    "Every trained model has exactly 20 evaluations",
)


duplicates = raw.duplicated(
    subset=[
        "run_id",
        "experiment",
        "environment",
    ]
).sum()


require(
    duplicates == 0,
    "No duplicate model/environment evaluations",
)


require(
    set(raw["dataset_id"].unique())
    == set(DATASETS),
    "All D00-D09 compositions are present",
)


require(
    set(raw["augmentation_id"].unique())
    == set(AUGMENTATIONS),
    "All A0-A4 augmentations are present",
)


require(
    set(raw["model_config"].unique())
    == set(MODELS),
    "All M0-M2 model configurations are present",
)


require(
    set(raw["seed"].astype(int).unique())
    == set(SEEDS),
    "All 10 independent confirmation seeds are present",
)


# ============================================================
# EXP1C
# ============================================================

section(
    "3. EXP1C TRAJECTORY STRUCTURE"
)


exp1c = raw.loc[
    raw["experiment"] == "EXP1C"
].copy()


require(
    len(exp1c) == 19500,
    "EXP1C contains exactly 19,500 rows",
)


temperature = exp1c.loc[
    exp1c["shift_type"]
    == "temperature"
]


humidity = exp1c.loc[
    exp1c["shift_type"]
    == "humidity"
]


require(
    len(temperature) == 9000,
    "Temperature contains exactly 9,000 rows",
)


require(
    len(humidity) == 10500,
    "Humidity contains exactly 10,500 rows",
)


require(
    set(
        temperature[
            "shift_value"
        ].astype(int)
    )
    == {
        23, 25, 27,
        29, 31, 33,
    },
    "Temperature physical shift values are correct",
)


require(
    set(
        humidity[
            "shift_value"
        ].astype(int)
    )
    == {
        65, 70, 75, 80,
        85, 90, 92,
    },
    "Humidity physical shift values are correct",
)


# ============================================================
# AUC table
# ============================================================

section(
    "4. AUC TABLE"
)


expected_auc = (
    10
    * 5
    * 3
    * 10
    * 2
)


require(
    expected_auc == 3000,
    "Expected AUC row calculation = 3,000",
)


require(
    len(auc) == 3000,
    "phase5_auc.csv contains exactly 3,000 trajectories",
)


auc_key = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "seed",
    "shift_type",
]


require(
    not auc.duplicated(
        subset=auc_key
    ).any(),
    "AUC table has one row per D x A x M x seed x shift family",
)


require(
    set(auc["shift_type"].unique())
    == set(SHIFT_TYPES),
    "AUC contains temperature and humidity only",
)


require(
    finite(auc["auc_mse"]),
    "All MSE AUC values are finite",
)


require(
    finite(auc["auc_mae"]),
    "All MAE AUC values are finite",
)


temp_auc = auc.loc[
    auc["shift_type"]
    == "temperature"
]


hum_auc = auc.loc[
    auc["shift_type"]
    == "humidity"
]


require(
    (temp_auc["n_points"] == 6).all(),
    "Every temperature AUC uses exactly 6 severity points",
)


require(
    (hum_auc["n_points"] == 7).all(),
    "Every humidity AUC uses exactly 7 severity points",
)


# ============================================================
# Independently recompute ALL AUC values
# ============================================================

section(
    "5. INDEPENDENT AUC RECALCULATION"
)


recomputed_auc = []


group_cols = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "seed",
    "shift_type",
]


for keys, g in exp1c.groupby(
    group_cols
):

    g = g.sort_values(
        "shift_value"
    )

    x = (
        g["shift_value"]
        .to_numpy(
            dtype=float
        )
    )

    mse = (
        g["mse"]
        .to_numpy(
            dtype=float
        )
    )

    mae = (
        g["mae"]
        .to_numpy(
            dtype=float
        )
    )

    if hasattr(
        np,
        "trapezoid",
    ):

        auc_mse = np.trapezoid(
            mse,
            x,
        )

        auc_mae = np.trapezoid(
            mae,
            x,
        )

    else:

        auc_mse = np.trapz(
            mse,
            x,
        )

        auc_mae = np.trapz(
            mae,
            x,
        )

    recomputed_auc.append({

        "dataset_id":
            keys[0],

        "augmentation_id":
            keys[1],

        "model_config":
            keys[2],

        "seed":
            int(keys[3]),

        "shift_type":
            keys[4],

        "recalc_auc_mse":
            float(auc_mse),

        "recalc_auc_mae":
            float(auc_mae),
    })


recomputed_auc = pd.DataFrame(
    recomputed_auc
)


auc_check = auc.merge(
    recomputed_auc,
    on=auc_key,
    how="inner",
    validate="one_to_one",
)


require(
    len(auc_check) == 3000,
    "All 3,000 stored AUC trajectories matched raw data",
)


require(
    close_enough(
        auc_check["auc_mse"],
        auc_check["recalc_auc_mse"],
    ),
    "Stored MSE AUC values exactly reproduce from raw EXP1C data",
)


require(
    close_enough(
        auc_check["auc_mae"],
        auc_check["recalc_auc_mae"],
    ),
    "Stored MAE AUC values exactly reproduce from raw EXP1C data",
)


# ============================================================
# Delta AUC
# ============================================================

section(
    "6. PAIRED DELTA-AUC INTEGRITY"
)


expected_delta_auc = (
    10
    * 4
    * 3
    * 10
    * 2
)


require(
    expected_delta_auc == 2400,
    "Expected paired delta-AUC rows = 2,400",
)


require(
    len(delta_auc) == 2400,
    "phase5_delta_auc.csv contains exactly 2,400 rows",
)


require(
    "A0"
    not in set(
        delta_auc[
            "augmentation_id"
        ].unique()
    ),
    "A0 is not incorrectly treated as a zero-valued augmentation level",
)


require(
    set(
        delta_auc[
            "augmentation_id"
        ].unique()
    )
    == set(
        NON_REFERENCE_AUGMENTATIONS
    ),
    "Delta-AUC contains A1-A4 only",
)


delta_key = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "seed",
    "shift_type",
]


require(
    not delta_auc.duplicated(
        subset=delta_key
    ).any(),
    "No duplicate paired delta-AUC observations",
)


require(
    finite(
        delta_auc[
            "delta_auc_mse"
        ]
    ),
    "All delta-AUC MSE values are finite",
)


require(
    finite(
        delta_auc[
            "delta_auc_mae"
        ]
    ),
    "All delta-AUC MAE values are finite",
)


# ============================================================
# Independently verify delta-AUC arithmetic
# ============================================================

reference_auc = (

    auc.loc[
        auc["augmentation_id"]
        == "A0",
        [
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
            "auc_mse",
            "auc_mae",
        ],
    ]

    .rename(
        columns={
            "auc_mse":
                "check_auc_mse_A0",

            "auc_mae":
                "check_auc_mae_A0",
        }
    )
)


delta_check = delta_auc.merge(

    reference_auc,

    on=[
        "dataset_id",
        "model_config",
        "seed",
        "shift_type",
    ],

    how="left",

    validate="many_to_one",
)


require(
    not delta_check[
        "check_auc_mse_A0"
    ].isna().any(),
    "Every delta-AUC observation has a valid A0 reference",
)


expected_delta_mse = (
    delta_check["auc_mse"]
    - delta_check["check_auc_mse_A0"]
)


expected_delta_mae = (
    delta_check["auc_mae"]
    - delta_check["check_auc_mae_A0"]
)


require(
    close_enough(
        delta_check[
            "delta_auc_mse"
        ],
        expected_delta_mse,
    ),
    "Every delta-AUC MSE value equals AUC(A) - AUC(A0)",
)


require(
    close_enough(
        delta_check[
            "delta_auc_mae"
        ],
        expected_delta_mae,
    ),
    "Every delta-AUC MAE value equals AUC(A) - AUC(A0)",
)


# ============================================================
# Delta-AUC summaries
# ============================================================

section(
    "7. DELTA-AUC SUMMARY STRUCTURE"
)


expected_summary_rows = (
    2
    * 4
    * 3
)


require(
    len(delta_summary)
    == expected_summary_rows,
    "Delta-AUC summary contains exactly 24 A x M x shift-family cells",
)


expected_cell_rows = (
    2
    * 10
    * 4
    * 3
)


require(
    len(delta_cells)
    == expected_cell_rows,
    "Delta-AUC cell table contains exactly 240 D x A x M x shift-family cells",
)


require(
    (
        delta_cells[
            "n_seeds"
        ] == 10
    ).all(),
    "Every D x A x M trajectory cell contains all 10 seeds",
)


# ============================================================
# Pointwise delta-MSE
# ============================================================

section(
    "8. POINTWISE DELTA-MSE INTEGRITY"
)


expected_pointwise = (
    10
    * 4
    * 3
    * 10
    * 13
)


require(
    expected_pointwise == 15600,
    "Expected pointwise paired rows = 15,600",
)


require(
    len(pointwise) == 15600,
    "phase5_pointwise_delta.csv contains exactly 15,600 rows",
)


require(
    set(
        pointwise[
            "augmentation_id"
        ].unique()
    )
    == set(
        NON_REFERENCE_AUGMENTATIONS
    ),
    "Pointwise paired data contains A1-A4 only",
)


require(
    finite(
        pointwise[
            "delta_mse"
        ]
    ),
    "All pointwise delta-MSE values are finite",
)


require(
    finite(
        pointwise[
            "delta_mae"
        ]
    ),
    "All pointwise delta-MAE values are finite",
)


require(
    finite(
        pointwise[
            "severity_z"
        ]
    ),
    "All standardized severity values are finite",
)


# ============================================================
# Independently verify pointwise pairing
# ============================================================

point_reference = (

    exp1c.loc[
        exp1c[
            "augmentation_id"
        ] == "A0",
        [
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
            "severity",
            "shift_value",
            "mse",
            "mae",
        ],
    ]

    .rename(
        columns={
            "mse":
                "check_mse_A0",

            "mae":
                "check_mae_A0",
        }
    )
)


point_check = pointwise.merge(

    point_reference,

    on=[
        "dataset_id",
        "model_config",
        "seed",
        "shift_type",
        "severity",
        "shift_value",
    ],

    how="left",

    validate="many_to_one",
)


require(
    not point_check[
        "check_mse_A0"
    ].isna().any(),
    "Every pointwise observation has its correct A0 reference",
)


expected_point_delta_mse = (
    point_check["mse"]
    - point_check["check_mse_A0"]
)


expected_point_delta_mae = (
    point_check["mae"]
    - point_check["check_mae_A0"]
)


require(
    close_enough(
        point_check["delta_mse"],
        expected_point_delta_mse,
    ),
    "Every pointwise delta-MSE equals MSE(A) - MSE(A0)",
)


require(
    close_enough(
        point_check["delta_mae"],
        expected_point_delta_mae,
    ),
    "Every pointwise delta-MAE equals MAE(A) - MAE(A0)",
)


# ============================================================
# H5.5 primary output
# ============================================================

section(
    "9. H5.5 PRIMARY OUTPUT"
)


require(
    len(h55) == 2,
    "H5.5 contains exactly one temperature and one humidity test",
)


require(
    set(
        h55["shift_type"].unique()
    )
    == set(SHIFT_TYPES),
    "H5.5 contains both shift families",
)


for col in [
    "f_statistic",
    "p_value",
    "p_fdr",
    "p_bonferroni",
]:

    require(
        finite(
            h55[col]
        ),
        f"H5.5 {col} is finite",
    )


require(
    (
        (h55["p_value"] >= 0)
        &
        (h55["p_value"] <= 1)
    ).all(),
    "H5.5 raw p-values are within [0,1]",
)


require(
    (
        (h55["p_fdr"] >= 0)
        &
        (h55["p_fdr"] <= 1)
    ).all(),
    "H5.5 FDR-adjusted p-values are within [0,1]",
)


# ============================================================
# H5.5 permutation sensitivity output
# ============================================================

section(
    "10. H5.5 PERMUTATION OUTPUT"
)


require(
    len(h55_perm) == 2,
    "Permutation summary contains both shift families",
)


require(
    set(
        h55_perm[
            "shift_type"
        ].unique()
    )
    == set(SHIFT_TYPES),
    "Permutation summary contains temperature and humidity",
)


for col in [
    "observed_f",
    "permutation_p",
    "p_fdr",
    "p_bonferroni",
]:

    require(
        finite(
            h55_perm[col]
        ),
        f"Permutation {col} is finite",
    )


# ============================================================
# Check observed F agrees between primary and permutation file
# ============================================================

f_compare = h55[
    [
        "shift_type",
        "f_statistic",
    ]
].merge(

    h55_perm[
        [
            "shift_type",
            "observed_f",
        ]
    ],

    on="shift_type",

    validate="one_to_one",
)


require(
    close_enough(
        f_compare[
            "f_statistic"
        ],
        f_compare[
            "observed_f"
        ],
    ),
    "Observed H5.5 F statistics agree across primary and permutation outputs",
)


# ============================================================
# H5.6
# ============================================================

section(
    "11. H5.6 OUTPUT"
)


require(
    len(h56) == 2,
    "H5.6 contains exactly two shift-family tests",
)


require(
    set(
        h56[
            "shift_type"
        ].unique()
    )
    == set(SHIFT_TYPES),
    "H5.6 contains temperature and humidity",
)


for col in [
    "f_statistic",
    "p_value",
    "p_fdr",
    "p_bonferroni",
]:

    require(
        finite(
            h56[col]
        ),
        f"H5.6 {col} is finite",
    )


require(
    (
        (h56["p_value"] >= 0)
        &
        (h56["p_value"] <= 1)
    ).all(),
    "H5.6 p-values are within [0,1]",
)


# ============================================================
# Per-severity output
# ============================================================

section(
    "12. PER-SEVERITY D x A x M OUTPUT"
)


require(
    len(per_severity) == 13,
    "Per-severity analysis contains exactly 13 severity tests",
)


temp_sev = per_severity.loc[
    per_severity["shift_type"]
    == "temperature"
]


hum_sev = per_severity.loc[
    per_severity["shift_type"]
    == "humidity"
]


require(
    len(temp_sev) == 6,
    "Temperature has exactly 6 per-severity tests",
)


require(
    len(hum_sev) == 7,
    "Humidity has exactly 7 per-severity tests",
)


require(
    set(
        temp_sev[
            "shift_value"
        ].astype(int)
    )
    == {
        23, 25, 27,
        29, 31, 33,
    },
    "Per-severity temperature values are correct",
)


require(
    set(
        hum_sev[
            "shift_value"
        ].astype(int)
    )
    == {
        65, 70, 75, 80,
        85, 90, 92,
    },
    "Per-severity humidity values are correct",
)


for col in [
    "f_statistic",
    "p_value",
    "p_fdr",
    "p_bonferroni",
]:

    require(
        finite(
            per_severity[col]
        ),
        f"Per-severity {col} is finite",
    )


# ============================================================
# Severity summary
# ============================================================

section(
    "13. SEVERITY SUMMARY"
)


# Temperature:
# 6 severity levels × 4 augmentations × 3 models = 72
#
# Humidity:
# 7 × 4 × 3 = 84
#
# Total = 156

require(
    len(severity_summary) == 156,
    "Severity summary contains exactly 156 cells",
)


require(
    finite(
        severity_summary[
            "mean_delta_mse"
        ]
    ),
    "Severity-summary means are finite",
)


# ============================================================
# Final result
# ============================================================

section(
    "FINAL RESULT"
)


print(
    f"Total integrity checks passed: "
    f"{checks_passed}"
)

print()

print(
    "PHASE 5 ANALYSIS INTEGRITY: VERIFIED PASS"
)

print()

print(
    "The analysis outputs are structurally and "
    "arithmetically consistent with the frozen "
    "30,000-record Phase 5 confirmation dataset."
)

print()

print(
    "NOTE:"
)

print(
    "This integrity check validates computation and "
    "structure. It does not by itself establish that "
    "every statistical inference method is the correct "
    "scientific test for the hypothesis."
)