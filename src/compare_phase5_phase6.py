"""
Phase 5 (G1) vs Phase 6 (G2) Cross-Generator Comparison
========================================================

Purpose
-------
Test how well augmentation-effect patterns observed under G1
generalize to the independently specified G2 generator.

NO TRAINING.
NO TUNING.
NO MODEL SELECTION.

Primary comparison units
------------------------
Trajectory level:
    D x A x M x shift family

Pointwise level:
    D x A x M x shift family x physical shift value

Effects
-------
Trajectory:
    mean paired delta AUC(MSE), A_k - A0

Pointwise:
    mean paired delta MSE, A_k - A0

Negative values:
    augmentation reduces error relative to A0.

Positive values:
    augmentation increases error relative to A0.

Replication is evaluated using:
1. sign agreement
2. Pearson correlation
3. Spearman correlation
4. magnitude agreement
5. severity-specific agreement
6. shift-family-specific agreement

This script deliberately does NOT define replication as
"both p-values < 0.05".
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

P5 = (
    ROOT
    / "results"
    / "experiment_5"
    / "confirmation"
    / "analysis"
)

P6 = (
    ROOT
    / "results"
    / "experiment_6"
    / "confirmation"
    / "analysis"
)

OUTPUT = (
    ROOT
    / "results"
    / "experiment_6"
    / "confirmation"
    / "analysis"
    / "g1_g2_comparison"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

P5_DELTA_AUC = (
    P5 / "phase5_delta_auc.csv"
)

P5_POINTWISE = (
    P5 / "phase5_pointwise_delta.csv"
)

P6_DELTA_AUC = (
    P6 / "phase6_delta_auc.csv"
)

P6_POINTWISE = (
    P6 / "phase6_pointwise_delta_mse.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

TRAJECTORY_MATCHED_FILE = (
    OUTPUT
    / "matched_trajectory_effects.csv"
)

TRAJECTORY_SUMMARY_FILE = (
    OUTPUT
    / "trajectory_replication_summary.csv"
)

POINTWISE_MATCHED_FILE = (
    OUTPUT
    / "matched_pointwise_effects.csv"
)

POINTWISE_SUMMARY_FILE = (
    OUTPUT
    / "pointwise_replication_summary.csv"
)

SEVERITY_SUMMARY_FILE = (
    OUTPUT
    / "severity_replication_summary.csv"
)

SIGN_MATRIX_FILE = (
    OUTPUT
    / "trajectory_sign_agreement.csv"
)

REPORT_FILE = (
    OUTPUT
    / "phase5_phase6_comparison_report.txt"
)

CONFIG_FILE = (
    OUTPUT
    / "comparison_config.json"
)


# ============================================================
# FROZEN FACTORS
# ============================================================

DATASETS = [
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
]

AUGMENTATIONS = [
    "A1", "A2", "A3", "A4",
]

MODELS = [
    "M0", "M1", "M2",
]

SHIFTS = [
    "temperature",
    "humidity",
]


# ============================================================
# HELPERS
# ============================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def safe_pearson(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    r, p = pearsonr(x, y)

    return float(r), float(p)


def safe_spearman(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return np.nan, np.nan

    r, p = spearmanr(x, y)

    return float(r), float(p)


def sign_agreement(x, y):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    sx = np.sign(x)
    sy = np.sign(y)

    return float(
        np.mean(sx == sy)
    )


def mean_absolute_difference(x, y):

    return float(
        np.mean(
            np.abs(
                np.asarray(x, dtype=float)
                - np.asarray(y, dtype=float)
            )
        )
    )


def rmse_difference(x, y):

    diff = (
        np.asarray(x, dtype=float)
        - np.asarray(y, dtype=float)
    )

    return float(
        np.sqrt(
            np.mean(diff ** 2)
        )
    )


def summarize_agreement(df, g1_col, g2_col):

    x = df[g1_col].to_numpy(dtype=float)
    y = df[g2_col].to_numpy(dtype=float)

    pearson_r, pearson_p = safe_pearson(x, y)
    spearman_r, spearman_p = safe_spearman(x, y)

    return {
        "n_cells":
            len(df),

        "pearson_r":
            pearson_r,

        "pearson_p":
            pearson_p,

        "spearman_r":
            spearman_r,

        "spearman_p":
            spearman_p,

        "sign_agreement":
            sign_agreement(x, y),

        "mean_g1":
            float(np.mean(x)),

        "mean_g2":
            float(np.mean(y)),

        "median_g1":
            float(np.median(x)),

        "median_g2":
            float(np.median(y)),

        "mean_absolute_difference":
            mean_absolute_difference(x, y),

        "rmse_difference":
            rmse_difference(x, y),

        "fraction_g1_improved":
            float(np.mean(x < 0)),

        "fraction_g2_improved":
            float(np.mean(y < 0)),
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print("PHASE 5 G1 vs PHASE 6 G2")
print("CROSS-GENERATOR REPLICATION ANALYSIS")
print("=" * 78)


for path in [
    P5_DELTA_AUC,
    P5_POINTWISE,
    P6_DELTA_AUC,
    P6_POINTWISE,
]:

    require(
        path.exists(),
        f"Missing required file: {path}",
    )


g1_auc = pd.read_csv(
    P5_DELTA_AUC
)

g2_auc = pd.read_csv(
    P6_DELTA_AUC
)

g1_point = pd.read_csv(
    P5_POINTWISE
)

g2_point = pd.read_csv(
    P6_POINTWISE
)


print()
print(
    "G1 delta-AUC rows:",
    len(g1_auc),
)

print(
    "G2 delta-AUC rows:",
    len(g2_auc),
)

print(
    "G1 pointwise rows:",
    len(g1_point),
)

print(
    "G2 pointwise rows:",
    len(g2_point),
)


# ============================================================
# BASIC INTEGRITY
# ============================================================

require(
    len(g1_auc) == 2400,
    f"Expected 2400 G1 delta-AUC rows, found {len(g1_auc)}.",
)

require(
    len(g2_auc) == 2400,
    f"Expected 2400 G2 delta-AUC rows, found {len(g2_auc)}.",
)

require(
    len(g1_point) == 15600,
    f"Expected 15600 G1 pointwise rows, found {len(g1_point)}.",
)

require(
    len(g2_point) == 15600,
    f"Expected 15600 G2 pointwise rows, found {len(g2_point)}.",
)


# ============================================================
# IMPORTANT:
#
# G1 and G2 use DIFFERENT confirmation seeds.
#
# Therefore DO NOT pair G1 seed 121 with G2 seed 301, etc.
#
# First average within each matched experimental cell across
# the independent seeds, and THEN compare G1 vs G2.
# ============================================================


# ============================================================
# TRAJECTORY-LEVEL CELL EFFECTS
#
# Expected:
#
# 10 D x 4 A x 3 M x 2 shifts = 240 cells
# ============================================================

trajectory_keys = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "shift_type",
]


g1_traj = (

    g1_auc

    .groupby(
        trajectory_keys,
        as_index=False,
    )

    .agg(
        g1_mean_delta_auc_mse=(
            "delta_auc_mse",
            "mean",
        ),

        g1_median_delta_auc_mse=(
            "delta_auc_mse",
            "median",
        ),

        g1_sd_delta_auc_mse=(
            "delta_auc_mse",
            "std",
        ),

        g1_n_seeds=(
            "seed",
            "nunique",
        ),
    )
)


g2_traj = (

    g2_auc

    .groupby(
        trajectory_keys,
        as_index=False,
    )

    .agg(
        g2_mean_delta_auc_mse=(
            "delta_auc_mse",
            "mean",
        ),

        g2_median_delta_auc_mse=(
            "delta_auc_mse",
            "median",
        ),

        g2_sd_delta_auc_mse=(
            "delta_auc_mse",
            "std",
        ),

        g2_n_seeds=(
            "seed",
            "nunique",
        ),
    )
)


require(
    len(g1_traj) == 240,
    f"Expected 240 G1 trajectory cells, found {len(g1_traj)}.",
)

require(
    len(g2_traj) == 240,
    f"Expected 240 G2 trajectory cells, found {len(g2_traj)}.",
)


require(
    (g1_traj["g1_n_seeds"] == 10).all(),
    "Some G1 trajectory cells do not contain 10 seeds.",
)

require(
    (g2_traj["g2_n_seeds"] == 10).all(),
    "Some G2 trajectory cells do not contain 10 seeds.",
)


matched_traj = g1_traj.merge(

    g2_traj,

    on=trajectory_keys,

    how="inner",

    validate="one_to_one",
)


require(
    len(matched_traj) == 240,
    (
        "Expected 240 matched G1/G2 trajectory cells; "
        f"found {len(matched_traj)}."
    ),
)


matched_traj[
    "g1_sign"
] = np.sign(
    matched_traj[
        "g1_mean_delta_auc_mse"
    ]
)


matched_traj[
    "g2_sign"
] = np.sign(
    matched_traj[
        "g2_mean_delta_auc_mse"
    ]
)


matched_traj[
    "same_direction"
] = (

    matched_traj["g1_sign"]
    == matched_traj["g2_sign"]
)


matched_traj[
    "absolute_effect_difference"
] = np.abs(

    matched_traj[
        "g1_mean_delta_auc_mse"
    ]

    - matched_traj[
        "g2_mean_delta_auc_mse"
    ]
)


matched_traj.to_csv(
    TRAJECTORY_MATCHED_FILE,
    index=False,
)


# ============================================================
# TRAJECTORY REPLICATION SUMMARY
# ============================================================

trajectory_summary = []


# Overall

result = summarize_agreement(

    matched_traj,

    "g1_mean_delta_auc_mse",

    "g2_mean_delta_auc_mse",
)

result[
    "scope"
] = "overall"

result[
    "shift_type"
] = "all"

trajectory_summary.append(
    result
)


# Shift-specific

for shift in SHIFTS:

    sub = matched_traj.loc[
        matched_traj["shift_type"]
        == shift
    ]

    require(
        len(sub) == 120,
        (
            f"Expected 120 {shift} trajectory cells, "
            f"found {len(sub)}."
        ),
    )

    result = summarize_agreement(

        sub,

        "g1_mean_delta_auc_mse",

        "g2_mean_delta_auc_mse",
    )

    result["scope"] = "shift_family"
    result["shift_type"] = shift

    trajectory_summary.append(
        result
    )


trajectory_summary = pd.DataFrame(
    trajectory_summary
)


trajectory_summary = trajectory_summary[
    [
        "scope",
        "shift_type",
        "n_cells",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "sign_agreement",
        "mean_g1",
        "mean_g2",
        "median_g1",
        "median_g2",
        "mean_absolute_difference",
        "rmse_difference",
        "fraction_g1_improved",
        "fraction_g2_improved",
    ]
]


trajectory_summary.to_csv(
    TRAJECTORY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# TRAJECTORY SIGN AGREEMENT BY A x M x SHIFT
#
# This gives a descriptive view of whether augmentation/model
# effect direction generalizes across training compositions.
# ============================================================

sign_matrix = (

    matched_traj

    .groupby(
        [
            "shift_type",
            "augmentation_id",
            "model_config",
        ],
        as_index=False,
    )

    .agg(
        n_datasets=(
            "dataset_id",
            "nunique",
        ),

        fraction_same_direction=(
            "same_direction",
            "mean",
        ),

        mean_g1_effect=(
            "g1_mean_delta_auc_mse",
            "mean",
        ),

        mean_g2_effect=(
            "g2_mean_delta_auc_mse",
            "mean",
        ),
    )
)


require(
    len(sign_matrix) == 24,
    (
        "Expected 24 A x M x shift sign-summary cells; "
        f"found {len(sign_matrix)}."
    ),
)


sign_matrix.to_csv(
    SIGN_MATRIX_FILE,
    index=False,
)


# ============================================================
# POINTWISE CELL EFFECTS
#
# Independent seeds again mean:
# average within D x A x M x shift x physical V first.
#
# Expected:
#
# Temperature:
# 10*4*3*6 = 720
#
# Humidity:
# 10*4*3*7 = 840
#
# Total = 1560 matched physical cells.
# ============================================================

point_keys = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "shift_type",
    "shift_value",
]


g1_point_cell = (

    g1_point

    .groupby(
        point_keys,
        as_index=False,
    )

    .agg(
        g1_mean_delta_mse=(
            "delta_mse",
            "mean",
        ),

        g1_median_delta_mse=(
            "delta_mse",
            "median",
        ),

        g1_sd_delta_mse=(
            "delta_mse",
            "std",
        ),

        g1_n_seeds=(
            "seed",
            "nunique",
        ),
    )
)


g2_point_cell = (

    g2_point

    .groupby(
        point_keys,
        as_index=False,
    )

    .agg(
        g2_mean_delta_mse=(
            "delta_mse",
            "mean",
        ),

        g2_median_delta_mse=(
            "delta_mse",
            "median",
        ),

        g2_sd_delta_mse=(
            "delta_mse",
            "std",
        ),

        g2_n_seeds=(
            "seed",
            "nunique",
        ),
    )
)


require(
    len(g1_point_cell) == 1560,
    (
        "Expected 1560 G1 pointwise cells; "
        f"found {len(g1_point_cell)}."
    ),
)

require(
    len(g2_point_cell) == 1560,
    (
        "Expected 1560 G2 pointwise cells; "
        f"found {len(g2_point_cell)}."
    ),
)


require(
    (g1_point_cell["g1_n_seeds"] == 10).all(),
    "Some G1 pointwise cells do not contain 10 seeds.",
)

require(
    (g2_point_cell["g2_n_seeds"] == 10).all(),
    "Some G2 pointwise cells do not contain 10 seeds.",
)


matched_point = g1_point_cell.merge(

    g2_point_cell,

    on=point_keys,

    how="inner",

    validate="one_to_one",
)


require(
    len(matched_point) == 1560,
    (
        "Expected 1560 matched pointwise cells; "
        f"found {len(matched_point)}."
    ),
)


matched_point[
    "same_direction"
] = (

    np.sign(
        matched_point[
            "g1_mean_delta_mse"
        ]
    )

    ==

    np.sign(
        matched_point[
            "g2_mean_delta_mse"
        ]
    )
)


matched_point[
    "absolute_effect_difference"
] = np.abs(

    matched_point[
        "g1_mean_delta_mse"
    ]

    - matched_point[
        "g2_mean_delta_mse"
    ]
)


matched_point.to_csv(
    POINTWISE_MATCHED_FILE,
    index=False,
)


# ============================================================
# POINTWISE SUMMARY
# ============================================================

pointwise_summary = []


result = summarize_agreement(

    matched_point,

    "g1_mean_delta_mse",

    "g2_mean_delta_mse",
)

result["scope"] = "overall"
result["shift_type"] = "all"

pointwise_summary.append(
    result
)


for shift in SHIFTS:

    sub = matched_point.loc[
        matched_point["shift_type"]
        == shift
    ]

    expected = (
        720
        if shift == "temperature"
        else 840
    )

    require(
        len(sub) == expected,
        (
            f"Expected {expected} pointwise cells "
            f"for {shift}; found {len(sub)}."
        ),
    )

    result = summarize_agreement(

        sub,

        "g1_mean_delta_mse",

        "g2_mean_delta_mse",
    )

    result["scope"] = "shift_family"
    result["shift_type"] = shift

    pointwise_summary.append(
        result
    )


pointwise_summary = pd.DataFrame(
    pointwise_summary
)


pointwise_summary = pointwise_summary[
    [
        "scope",
        "shift_type",
        "n_cells",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "sign_agreement",
        "mean_g1",
        "mean_g2",
        "median_g1",
        "median_g2",
        "mean_absolute_difference",
        "rmse_difference",
        "fraction_g1_improved",
        "fraction_g2_improved",
    ]
]


pointwise_summary.to_csv(
    POINTWISE_SUMMARY_FILE,
    index=False,
)


# ============================================================
# PHYSICAL-SEVERITY REPLICATION
#
# One row for each of:
#
# temperature: 23,25,27,29,31,33
# humidity:    65,70,75,80,85,90,92
#
# Each physical severity has:
#
# 10 D * 4 A * 3 M = 120 cells.
# ============================================================

severity_summary = []


for shift in SHIFTS:

    shift_df = matched_point.loc[
        matched_point["shift_type"]
        == shift
    ]


    for value in sorted(
        shift_df["shift_value"].unique()
    ):

        sub = shift_df.loc[
            np.isclose(
                shift_df[
                    "shift_value"
                ].astype(float),
                float(value),
            )
        ]


        require(
            len(sub) == 120,
            (
                f"Expected 120 cells for {shift} "
                f"at {value}; found {len(sub)}."
            ),
        )


        result = summarize_agreement(

            sub,

            "g1_mean_delta_mse",

            "g2_mean_delta_mse",
        )


        result[
            "shift_type"
        ] = shift

        result[
            "shift_value"
        ] = float(value)


        severity_summary.append(
            result
        )


severity_summary = pd.DataFrame(
    severity_summary
)


require(
    len(severity_summary) == 13,
    (
        "Expected 13 physical-severity comparison rows; "
        f"found {len(severity_summary)}."
    ),
)


severity_summary = severity_summary[
    [
        "shift_type",
        "shift_value",
        "n_cells",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "sign_agreement",
        "mean_g1",
        "mean_g2",
        "median_g1",
        "median_g2",
        "mean_absolute_difference",
        "rmse_difference",
        "fraction_g1_improved",
        "fraction_g2_improved",
    ]
]


severity_summary.to_csv(
    SEVERITY_SUMMARY_FILE,
    index=False,
)


# ============================================================
# CROSS-GENERATOR EFFECT-SCALE DIAGNOSTIC
#
# We report effect-scale ratios descriptively, but do not use
# them as a replication pass/fail criterion.
# ============================================================

for frame in [
    trajectory_summary,
    pointwise_summary,
    severity_summary,
]:

    pass


# ============================================================
# CONFIG
# ============================================================

CONFIG = {

    "analysis":
        "Phase5_G1_vs_Phase6_G2",

    "comparison_status":
        "POST_CONFIRMATION_CROSS_GENERATOR",

    "g1":
        "Phase 5 frozen G1 confirmation",

    "g2":
        "Phase 6 frozen G2 confirmation",

    "trajectory_unit":
        "D x A x M x shift_family",

    "pointwise_unit":
        (
            "D x A x M x shift_family "
            "x physical_shift_value"
        ),

    "trajectory_cells":
        240,

    "pointwise_cells":
        1560,

    "important_seed_rule":
        (
            "G1 and G2 have independent confirmation seeds. "
            "Effects are averaged within matched experimental "
            "cells before cross-generator comparison. "
            "Seeds are not paired across generators."
        ),

    "replication_dimensions": [
        "effect direction",
        "effect magnitude",
        "Pearson correlation",
        "Spearman rank correlation",
        "physical severity pattern",
        "shift-family dependence",
    ],

    "interpretation_rule":
        (
            "Replication is multidimensional and must not "
            "be reduced to whether both generators have "
            "p < 0.05."
        ),
}


with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        CONFIG,
        f,
        indent=2,
    )


# ============================================================
# REPORT
# ============================================================

report = []

report.append(
    "PHASE 5 G1 vs PHASE 6 G2"
)

report.append(
    "CROSS-GENERATOR REPLICATION ANALYSIS"
)

report.append(
    "=" * 70
)

report.append("")

report.append(
    "INTEGRITY"
)

report.append(
    f"G1 trajectory seed-level rows: {len(g1_auc)}"
)

report.append(
    f"G2 trajectory seed-level rows: {len(g2_auc)}"
)

report.append(
    f"Matched trajectory cells: {len(matched_traj)}"
)

report.append(
    f"Matched pointwise cells: {len(matched_point)}"
)

report.append("")

report.append(
    "IMPORTANT:"
)

report.append(
    (
        "G1 and G2 use independent confirmation seeds. "
        "Seeds were therefore not paired across generators. "
        "Seed-level effects were first averaged within each "
        "matched experimental cell."
    )
)

report.append("")

report.append(
    "TRAJECTORY-LEVEL AGREEMENT"
)

report.append(
    trajectory_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    "POINTWISE AGREEMENT"
)

report.append(
    pointwise_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    "PHYSICAL-SEVERITY AGREEMENT"
)

report.append(
    severity_summary.to_string(
        index=False
    )
)

report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    (
        "These statistics describe cross-generator "
        "agreement in augmentation effects. They should "
        "be interpreted jointly rather than using any "
        "single correlation, sign-agreement percentage, "
        "or p-value as a binary replication criterion."
    )
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(report)
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 78)
print("TRAJECTORY-LEVEL CROSS-GENERATOR AGREEMENT")
print("=" * 78)

print(
    trajectory_summary.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("POINTWISE CROSS-GENERATOR AGREEMENT")
print("=" * 78)

print(
    pointwise_summary.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("PHYSICAL-SEVERITY AGREEMENT")
print("=" * 78)

print(
    severity_summary[
        [
            "shift_type",
            "shift_value",
            "pearson_r",
            "spearman_r",
            "sign_agreement",
            "mean_g1",
            "mean_g2",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FINAL INTEGRITY ASSERTIONS
# ============================================================

require(
    len(matched_traj) == 240,
    "Final trajectory-cell count failed.",
)

require(
    len(matched_point) == 1560,
    "Final pointwise-cell count failed.",
)

require(
    len(trajectory_summary) == 3,
    "Trajectory summary should contain overall/temp/humidity.",
)

require(
    len(pointwise_summary) == 3,
    "Pointwise summary should contain overall/temp/humidity.",
)

require(
    len(severity_summary) == 13,
    "Severity summary should contain 13 rows.",
)

require(
    len(sign_matrix) == 24,
    "Sign matrix should contain 24 rows.",
)


for column in [
    "g1_mean_delta_auc_mse",
    "g2_mean_delta_auc_mse",
]:

    require(
        np.isfinite(
            matched_traj[column]
        ).all(),
        f"Non-finite trajectory effects: {column}",
    )


for column in [
    "g1_mean_delta_mse",
    "g2_mean_delta_mse",
]:

    require(
        np.isfinite(
            matched_point[column]
        ).all(),
        f"Non-finite pointwise effects: {column}",
    )


# Every expected D/A/M/shift trajectory cell must exist

expected_trajectory_cells = (
    len(DATASETS)
    * len(AUGMENTATIONS)
    * len(MODELS)
    * len(SHIFTS)
)

require(
    expected_trajectory_cells == 240,
    "Internal expected trajectory count incorrect.",
)


# ============================================================
# FINAL PASS
# ============================================================

print()
print("=" * 78)

print(
    "PHASE 5 vs PHASE 6 CROSS-GENERATOR "
    "COMPARISON: COMPUTATIONAL PASS"
)

print("=" * 78)

print()

print(
    f"Matched trajectory cells: {len(matched_traj)}"
)

print(
    f"Matched pointwise cells:  {len(matched_point)}"
)

print(
    f"Severity comparisons:     {len(severity_summary)}"
)

print()

print(
    "No retraining or generator modification was performed."
)

print(
    "Cross-generator seeds were not incorrectly paired."
)

print()

print(
    f"Results written to:\n{OUTPUT}"
)