"""
Phase 6 Confirmation Analysis
=============================

Purpose
-------
Analyze the frozen Phase 6 G2 confirmation experiment using the
same primary statistical framework used for the final frozen
Phase 5 analysis.

NO TRAINING IS PERFORMED HERE.

Input
-----
results/experiment_6/confirmation/master_results.csv

Primary analyses
----------------
H6.5:
    D x A x M interaction on integrated degradation trajectories.

    Outcome:
        paired delta AUC(MSE) relative to A0

    Analysis:
        temperature and humidity separately

    Primary test:
        fixed-block nested partial-F test

H6.6:
    D x A x M x V interaction.

    Outcome:
        pointwise paired delta MSE relative to A0

    Analysis:
        temperature and humidity separately

    Primary test:
        fixed-block nested partial-F test with standardized
        continuous severity.

Secondary:
    per-severity D x A x M nested F tests

Important
---------
The Phase 5 augmentation-identity permutation sensitivity test is
NOT used as the primary H6.5 test because it represented a broader
augmentation-exchangeability null rather than the conditional
three-way interaction null.

This script does not:
- retrain models
- change G2
- change seeds
- select cells
- remove difficult results
- tune augmentations
- tune architectures
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "experiment_6"
    / "confirmation"
    / "master_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiment_6"
    / "confirmation"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

AUC_FILE = (
    OUTPUT_DIR
    / "phase6_auc.csv"
)

DELTA_AUC_FILE = (
    OUTPUT_DIR
    / "phase6_delta_auc.csv"
)

AUC_SUMMARY_FILE = (
    OUTPUT_DIR
    / "phase6_delta_auc_summary.csv"
)

POINTWISE_FILE = (
    OUTPUT_DIR
    / "phase6_pointwise_delta_mse.csv"
)

H65_FILE = (
    OUTPUT_DIR
    / "phase6_h65_dam.csv"
)

H66_FILE = (
    OUTPUT_DIR
    / "phase6_h66_damv.csv"
)

PER_SEVERITY_FILE = (
    OUTPUT_DIR
    / "phase6_per_severity_dam.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "phase6_confirmation_report.txt"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "analysis_config.json"
)


# ============================================================
# FROZEN FACTORS
# ============================================================

DATASETS = [
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
]

AUGMENTATIONS = [
    "A0", "A1", "A2", "A3", "A4",
]

NONBASE_AUGMENTATIONS = [
    "A1", "A2", "A3", "A4",
]

MODELS = [
    "M0", "M1", "M2",
]

SEEDS = [
    301, 312, 323, 334, 345,
    356, 367, 378, 389, 400,
]

SHIFT_FAMILIES = [
    "temperature",
    "humidity",
]


# ============================================================
# ANALYSIS CONFIGURATION
# ============================================================

ANALYSIS_CONFIG = {

    "phase": "PHASE6",

    "generator": "G2",

    "analysis_status":
        "CONFIRMATORY_FROZEN_DATA",

    "baseline_augmentation":
        "A0",

    "trajectory_metric":
        "trapezoidal AUC of MSE over physical shift value",

    "paired_trajectory_effect":
        "A_k AUC - A0 AUC",

    "pointwise_effect":
        "A_k MSE - A0 MSE",

    "primary_h65":
        (
            "fixed-block nested partial-F test of "
            "D x A x M on paired delta AUC"
        ),

    "primary_h66":
        (
            "fixed-block nested partial-F test of "
            "D x A x M x severity_z on paired "
            "pointwise delta MSE"
        ),

    "seed_treatment":
        "seed included as fixed block",

    "shift_families":
        SHIFT_FAMILIES,

    "multiple_testing":
        "Benjamini-Hochberg FDR across shift families",

    "per_severity_status":
        "secondary sensitivity analysis",
}


# ============================================================
# HELPERS
# ============================================================

def require(condition, message):

    if not condition:

        raise RuntimeError(
            message
        )


def trapezoid_auc(x, y):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # numpy 2.x prefers trapezoid;
    # older numpy uses trapz.

    if hasattr(
        np,
        "trapezoid",
    ):

        return float(
            np.trapezoid(
                y,
                x,
            )
        )

    return float(
        np.trapz(
            y,
            x,
        )
    )


def nested_partial_f(
    data,
    reduced_formula,
    full_formula,
):

    """
    Fit reduced and full OLS models to the SAME rows and perform
    the classical nested partial-F comparison.

    Seed is explicitly included in formulas as a fixed block.
    """

    reduced = smf.ols(
        reduced_formula,
        data=data,
    ).fit()

    full = smf.ols(
        full_formula,
        data=data,
    ).fit()

    require(
        int(reduced.nobs)
        == int(full.nobs),
        "Reduced and full models used different observations.",
    )

    f_stat, p_value, df_diff = (
        full.compare_f_test(
            reduced
        )
    )

    return {
        "f_stat":
            float(f_stat),

        "p_value":
            float(p_value),

        "df_num":
            int(df_diff),

        "df_resid_full":
            int(full.df_resid),

        "n_obs":
            int(full.nobs),

        "rss_reduced":
            float(
                np.sum(
                    reduced.resid ** 2
                )
            ),

        "rss_full":
            float(
                np.sum(
                    full.resid ** 2
                )
            ),
    }


# ============================================================
# LOAD RAW CONFIRMATION DATA
# ============================================================

print("=" * 78)
print("PHASE 6 CONFIRMATION ANALYSIS")
print("=" * 78)

require(
    INPUT_FILE.exists(),
    f"Missing input file: {INPUT_FILE}",
)

raw = pd.read_csv(
    INPUT_FILE
)

print(
    f"Raw rows: {len(raw)}"
)


# ============================================================
# RAW DATA INTEGRITY
# ============================================================

require(
    len(raw) == 30000,
    (
        "Expected 30000 Phase 6 confirmation "
        f"records; found {len(raw)}."
    ),
)

require(
    raw["run_id"].nunique()
    == 1500,
    "Expected exactly 1500 unique trained models.",
)

require(
    set(raw["dataset_id"].unique())
    == set(DATASETS),
    "Dataset IDs do not match frozen Phase 6 design.",
)

require(
    set(raw["augmentation_id"].unique())
    == set(AUGMENTATIONS),
    "Augmentation IDs do not match frozen design.",
)

require(
    set(raw["model_config"].unique())
    == set(MODELS),
    "Model configurations do not match frozen design.",
)

require(
    set(
        raw["seed"]
        .astype(int)
        .unique()
    )
    == set(SEEDS),
    "Seeds do not match frozen Phase 6 confirmation seeds.",
)

require(
    raw[
        [
            "mse",
            "mae",
            "best_validation_loss",
        ]
    ]
    .notna()
    .all()
    .all(),
    "Missing numerical performance values.",
)

require(
    np.isfinite(
        raw[
            [
                "mse",
                "mae",
                "best_validation_loss",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all(),
    "Non-finite numerical performance values detected.",
)


# ============================================================
# EXTRACT EXP1C
# ============================================================

severity = raw.loc[
    raw["experiment"] == "EXP1C"
].copy()

require(
    len(severity) == 19500,
    (
        "Expected 19500 EXP1C rows; "
        f"found {len(severity)}."
    ),
)

require(
    len(
        severity.loc[
            severity["shift_type"]
            == "temperature"
        ]
    )
    == 9000,
    "Expected 9000 temperature severity records.",
)

require(
    len(
        severity.loc[
            severity["shift_type"]
            == "humidity"
        ]
    )
    == 10500,
    "Expected 10500 humidity severity records.",
)


# ============================================================
# CONSTRUCT TRAJECTORY AUC
#
# One AUC for:
#
# D x A x M x seed x shift_family
#
# Expected:
#
# 10 x 5 x 3 x 10 x 2 = 3000
# ============================================================

print()
print("Computing trajectory AUCs...")

auc_records = []

group_columns = [
    "dataset_id",
    "augmentation_id",
    "model_config",
    "seed",
    "shift_type",
]


for keys, group in severity.groupby(
    group_columns,
    sort=True,
):

    (
        dataset_id,
        augmentation_id,
        model_config,
        seed,
        shift_type,

    ) = keys


    expected_points = (

        6
        if shift_type == "temperature"
        else 7
    )


    require(
        len(group)
        == expected_points,
        (
            "Unexpected trajectory length for "
            f"{keys}: {len(group)}"
        ),
    )


    require(
        group["shift_value"].nunique()
        == expected_points,
        (
            "Duplicate/missing physical severity "
            f"points for {keys}."
        ),
    )


    x = group[
        "shift_value"
    ].to_numpy(
        dtype=float
    )

    mse = group[
        "mse"
    ].to_numpy(
        dtype=float
    )

    mae = group[
        "mae"
    ].to_numpy(
        dtype=float
    )


    auc_mse = trapezoid_auc(
        x,
        mse,
    )

    auc_mae = trapezoid_auc(
        x,
        mae,
    )


    auc_records.append({

        "phase":
            "PHASE6",

        "generator":
            "G2",

        "dataset_id":
            dataset_id,

        "augmentation_id":
            augmentation_id,

        "model_config":
            model_config,

        "seed":
            int(seed),

        "shift_type":
            shift_type,

        "n_points":
            expected_points,

        "min_shift_value":
            float(
                np.min(x)
            ),

        "max_shift_value":
            float(
                np.max(x)
            ),

        "auc_mse":
            auc_mse,

        "auc_mae":
            auc_mae,
    })


auc = pd.DataFrame(
    auc_records
)


require(
    len(auc) == 3000,
    (
        "Expected 3000 trajectory AUC rows; "
        f"found {len(auc)}."
    ),
)


require(
    not auc.duplicated(
        subset=[
            "dataset_id",
            "augmentation_id",
            "model_config",
            "seed",
            "shift_type",
        ]
    ).any(),
    "Duplicate trajectory AUC cells detected.",
)


auc.to_csv(
    AUC_FILE,
    index=False,
)

print(
    f"AUC rows: {len(auc)}"
)


# ============================================================
# CONSTRUCT PAIRED DELTA AUC
#
# delta = augmentation - A0
#
# Negative delta:
# augmentation reduces integrated error.
#
# Positive delta:
# augmentation increases integrated error.
#
# Expected:
#
# 10 D
# x 4 non-A0 A
# x 3 M
# x 10 seeds
# x 2 shift families
#
# = 2400
# ============================================================

print()
print("Computing paired delta AUCs...")

baseline_auc = (

    auc.loc[
        auc["augmentation_id"]
        == "A0"
    ]

    [
        [
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
            "auc_mse",
            "auc_mae",
        ]
    ]

    .rename(
        columns={
            "auc_mse":
                "auc_mse_a0",

            "auc_mae":
                "auc_mae_a0",
        }
    )
)


treated_auc = (

    auc.loc[
        auc["augmentation_id"]
        != "A0"
    ]

    .copy()
)


delta_auc = treated_auc.merge(

    baseline_auc,

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
    delta_auc[
        [
            "auc_mse_a0",
            "auc_mae_a0",
        ]
    ]
    .notna()
    .all()
    .all(),
    "Missing A0 trajectory pairing.",
)


delta_auc[
    "delta_auc_mse"
] = (

    delta_auc["auc_mse"]

    - delta_auc["auc_mse_a0"]
)


delta_auc[
    "delta_auc_mae"
] = (

    delta_auc["auc_mae"]

    - delta_auc["auc_mae_a0"]
)


require(
    len(delta_auc) == 2400,
    (
        "Expected 2400 paired delta AUC rows; "
        f"found {len(delta_auc)}."
    ),
)


require(
    not delta_auc.duplicated(
        subset=[
            "dataset_id",
            "augmentation_id",
            "model_config",
            "seed",
            "shift_type",
        ]
    ).any(),
    "Duplicate paired delta-AUC cells detected.",
)


delta_auc.to_csv(
    DELTA_AUC_FILE,
    index=False,
)

print(
    f"Delta-AUC rows: {len(delta_auc)}"
)


# ============================================================
# DELTA-AUC DESCRIPTIVE SUMMARY
# ============================================================

auc_summary = (

    delta_auc

    .groupby(
        [
            "shift_type",
            "dataset_id",
            "augmentation_id",
            "model_config",
        ],
        as_index=False,
    )

    .agg(

        n_seeds=(
            "seed",
            "nunique",
        ),

        mean_delta_auc_mse=(
            "delta_auc_mse",
            "mean",
        ),

        median_delta_auc_mse=(
            "delta_auc_mse",
            "median",
        ),

        sd_delta_auc_mse=(
            "delta_auc_mse",
            "std",
        ),

        min_delta_auc_mse=(
            "delta_auc_mse",
            "min",
        ),

        max_delta_auc_mse=(
            "delta_auc_mse",
            "max",
        ),

        mean_delta_auc_mae=(
            "delta_auc_mae",
            "mean",
        ),
    )
)


auc_summary[
    "fraction_seeds_improved"
] = (

    delta_auc

    .assign(
        improved=lambda x:
            x["delta_auc_mse"] < 0
    )

    .groupby(
        [
            "shift_type",
            "dataset_id",
            "augmentation_id",
            "model_config",
        ]
    )[
        "improved"
    ]

    .mean()

    .to_numpy()
)


require(
    len(auc_summary) == 240,
    (
        "Expected 240 trajectory-cell summary rows; "
        f"found {len(auc_summary)}."
    ),
)


auc_summary.to_csv(
    AUC_SUMMARY_FILE,
    index=False,
)


# ============================================================
# H6.5
#
# TRAJECTORY-LEVEL:
#
# D x A x M
#
# Outcome:
#     delta_auc_mse
#
# Seed is a fixed block.
#
# Reduced:
#     seed
#     + D*A
#     + D*M
#     + A*M
#
# Full:
#     seed
#     + D*A*M
#
# Difference = D:A:M terms.
#
# Temperature and humidity tested separately.
# ============================================================

print()
print("=" * 78)
print("H6.5 — TRAJECTORY D x A x M")
print("=" * 78)


h65_results = []


H65_REDUCED = (
    "delta_auc_mse ~ "
    "C(seed) + "
    "C(dataset_id) * C(augmentation_id) + "
    "C(dataset_id) * C(model_config) + "
    "C(augmentation_id) * C(model_config)"
)


H65_FULL = (
    "delta_auc_mse ~ "
    "C(seed) + "
    "C(dataset_id) * "
    "C(augmentation_id) * "
    "C(model_config)"
)


for shift_type in SHIFT_FAMILIES:

    df = (

        delta_auc.loc[
            delta_auc["shift_type"]
            == shift_type
        ]

        .copy()
    )


    require(
        len(df) == 1200,
        (
            f"H6.5 {shift_type}: "
            f"expected 1200 rows, found {len(df)}."
        ),
    )


    result = nested_partial_f(

        data=df,

        reduced_formula=
            H65_REDUCED,

        full_formula=
            H65_FULL,
    )


    result[
        "shift_type"
    ] = shift_type


    result[
        "hypothesis"
    ] = "H6.5"


    result[
        "effect_tested"
    ] = "D x A x M"


    h65_results.append(
        result
    )


h65 = pd.DataFrame(
    h65_results
)


# FDR across the two shift-family H6.5 tests

reject, p_fdr, _, _ = (
    multipletests(

        h65[
            "p_value"
        ],

        alpha=0.05,

        method="fdr_bh",
    )
)


h65[
    "p_fdr_bh"
] = p_fdr


h65[
    "reject_fdr_05"
] = reject


h65[
    "p_bonferroni"
] = np.minimum(

    h65[
        "p_value"
    ]
    * len(h65),

    1.0,
)


h65 = h65[
    [
        "hypothesis",
        "shift_type",
        "effect_tested",
        "n_obs",
        "df_num",
        "df_resid_full",
        "f_stat",
        "p_value",
        "p_fdr_bh",
        "p_bonferroni",
        "reject_fdr_05",
        "rss_reduced",
        "rss_full",
    ]
]


h65.to_csv(
    H65_FILE,
    index=False,
)


print(
    h65.to_string(
        index=False
    )
)


# ============================================================
# CONSTRUCT POINTWISE PAIRED DELTA MSE
#
# Match:
#
# D
# M
# seed
# shift family
# physical shift value
#
# delta MSE = A_k - A0
#
# Expected:
#
# temperature:
# 10*4*3*10*6 = 7200
#
# humidity:
# 10*4*3*10*7 = 8400
#
# total = 15600
# ============================================================

print()
print("Computing pointwise paired delta-MSE...")

baseline_point = (

    severity.loc[
        severity["augmentation_id"]
        == "A0"
    ]

    [
        [
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
            "severity",
            "shift_value",
            "mse",
            "mae",
        ]
    ]

    .rename(
        columns={
            "mse":
                "mse_a0",

            "mae":
                "mae_a0",
        }
    )
)


treated_point = (

    severity.loc[
        severity["augmentation_id"]
        != "A0"
    ]

    [
        [
            "dataset_id",
            "augmentation_id",
            "model_config",
            "seed",
            "shift_type",
            "severity",
            "shift_value",
            "mse",
            "mae",
        ]
    ]

    .copy()
)


pointwise = treated_point.merge(

    baseline_point,

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
    pointwise[
        [
            "mse_a0",
            "mae_a0",
        ]
    ]
    .notna()
    .all()
    .all(),
    "Missing A0 pointwise pairing.",
)


pointwise[
    "delta_mse"
] = (

    pointwise["mse"]

    - pointwise["mse_a0"]
)


pointwise[
    "delta_mae"
] = (

    pointwise["mae"]

    - pointwise["mae_a0"]
)


require(
    len(pointwise) == 15600,
    (
        "Expected 15600 pointwise deltas; "
        f"found {len(pointwise)}."
    ),
)


require(
    len(
        pointwise.loc[
            pointwise["shift_type"]
            == "temperature"
        ]
    )
    == 7200,
    "Expected 7200 temperature pointwise deltas.",
)


require(
    len(
        pointwise.loc[
            pointwise["shift_type"]
            == "humidity"
        ]
    )
    == 8400,
    "Expected 8400 humidity pointwise deltas.",
)


# ============================================================
# STANDARDIZE PHYSICAL SEVERITY WITHIN SHIFT FAMILY
#
# z-score is calculated from the physical shift-value grid.
#
# This keeps severity continuous for H6.6.
# ============================================================

pointwise[
    "severity_z"
] = np.nan


for shift_type in SHIFT_FAMILIES:

    mask = (
        pointwise["shift_type"]
        == shift_type
    )

    values = (

        pointwise.loc[
            mask,
            "shift_value",
        ]

        .astype(float)
    )


    mean_value = (
        values.mean()
    )

    sd_value = (
        values.std(
            ddof=0
        )
    )


    require(
        sd_value > 0,
        (
            f"Zero severity SD for "
            f"{shift_type}."
        ),
    )


    pointwise.loc[
        mask,
        "severity_z",
    ] = (

        values
        - mean_value

    ) / sd_value


require(
    pointwise[
        "severity_z"
    ]
    .notna()
    .all(),
    "severity_z contains missing values.",
)


pointwise.to_csv(
    POINTWISE_FILE,
    index=False,
)


print(
    f"Pointwise delta rows: {len(pointwise)}"
)


# ============================================================
# H6.6
#
# D x A x M x V
#
# Outcome:
#     pointwise delta MSE
#
# Full:
#
# seed + D*A*M*severity_z
#
# Reduced:
#
# retains:
#   D*A*M
#   D*A*V
#   D*M*V
#   A*M*V
#
# but excludes:
#   D*A*M*V
#
# Therefore the nested comparison isolates the four-way
# D x A x M x severity interaction.
# ============================================================

print()
print("=" * 78)
print("H6.6 — D x A x M x SEVERITY")
print("=" * 78)


H66_REDUCED = (
    "delta_mse ~ "
    "C(seed) + "
    "C(dataset_id) * C(augmentation_id) * C(model_config) + "
    "severity_z * "
    "("
    "C(dataset_id) * C(augmentation_id) + "
    "C(dataset_id) * C(model_config) + "
    "C(augmentation_id) * C(model_config)"
    ")"
)


H66_FULL = (
    "delta_mse ~ "
    "C(seed) + "
    "C(dataset_id) * "
    "C(augmentation_id) * "
    "C(model_config) * "
    "severity_z"
)


h66_results = []


for shift_type in SHIFT_FAMILIES:

    df = (

        pointwise.loc[
            pointwise["shift_type"]
            == shift_type
        ]

        .copy()
    )


    expected_n = (

        7200
        if shift_type == "temperature"
        else 8400
    )


    require(
        len(df) == expected_n,
        (
            f"H6.6 {shift_type}: "
            f"expected {expected_n}, found {len(df)}."
        ),
    )


    result = nested_partial_f(

        data=df,

        reduced_formula=
            H66_REDUCED,

        full_formula=
            H66_FULL,
    )


    result[
        "shift_type"
    ] = shift_type


    result[
        "hypothesis"
    ] = "H6.6"


    result[
        "effect_tested"
    ] = "D x A x M x V"


    h66_results.append(
        result
    )


h66 = pd.DataFrame(
    h66_results
)


reject, p_fdr, _, _ = (
    multipletests(

        h66[
            "p_value"
        ],

        alpha=0.05,

        method="fdr_bh",
    )
)


h66[
    "p_fdr_bh"
] = p_fdr


h66[
    "reject_fdr_05"
] = reject


h66[
    "p_bonferroni"
] = np.minimum(

    h66[
        "p_value"
    ]
    * len(h66),

    1.0,
)


h66 = h66[
    [
        "hypothesis",
        "shift_type",
        "effect_tested",
        "n_obs",
        "df_num",
        "df_resid_full",
        "f_stat",
        "p_value",
        "p_fdr_bh",
        "p_bonferroni",
        "reject_fdr_05",
        "rss_reduced",
        "rss_full",
    ]
]


h66.to_csv(
    H66_FILE,
    index=False,
)


print(
    h66.to_string(
        index=False
    )
)


# ============================================================
# SECONDARY PER-SEVERITY D x A x M
#
# At each physical severity:
#
# Reduced:
# seed + D*A + D*M + A*M
#
# Full:
# seed + D*A*M
#
# This is SECONDARY.
#
# It must not be interpreted as a preregistered threshold search.
# ============================================================

print()
print("=" * 78)
print("SECONDARY PER-SEVERITY D x A x M")
print("=" * 78)


per_severity_results = []


PER_SEVERITY_REDUCED = (
    "delta_mse ~ "
    "C(seed) + "
    "C(dataset_id) * C(augmentation_id) + "
    "C(dataset_id) * C(model_config) + "
    "C(augmentation_id) * C(model_config)"
)


PER_SEVERITY_FULL = (
    "delta_mse ~ "
    "C(seed) + "
    "C(dataset_id) * "
    "C(augmentation_id) * "
    "C(model_config)"
)


for shift_type in SHIFT_FAMILIES:

    shift_df = (

        pointwise.loc[
            pointwise["shift_type"]
            == shift_type
        ]

        .copy()
    )


    physical_values = sorted(

        shift_df[
            "shift_value"
        ]
        .astype(float)
        .unique()
    )


    for shift_value in physical_values:

        df = (

            shift_df.loc[

                np.isclose(
                    shift_df[
                        "shift_value"
                    ].astype(float),
                    shift_value,
                )

            ]

            .copy()
        )


        require(
            len(df) == 1200,
            (
                "Per-severity cell should contain "
                f"1200 rows: {shift_type}, "
                f"{shift_value}; found {len(df)}."
            ),
        )


        result = nested_partial_f(

            data=df,

            reduced_formula=
                PER_SEVERITY_REDUCED,

            full_formula=
                PER_SEVERITY_FULL,
        )


        result[
            "shift_type"
        ] = shift_type


        result[
            "shift_value"
        ] = float(
            shift_value
        )


        result[
            "effect_tested"
        ] = "D x A x M"


        per_severity_results.append(
            result
        )


per_severity = pd.DataFrame(
    per_severity_results
)


require(
    len(per_severity) == 13,
    (
        "Expected 13 per-severity tests; "
        f"found {len(per_severity)}."
    ),
)


# ------------------------------------------------------------
# FDR across all 13 secondary severity tests
# ------------------------------------------------------------

reject, p_fdr, _, _ = (
    multipletests(

        per_severity[
            "p_value"
        ],

        alpha=0.05,

        method="fdr_bh",
    )
)


per_severity[
    "p_fdr_bh"
] = p_fdr


per_severity[
    "reject_fdr_05"
] = reject


per_severity[
    "p_bonferroni"
] = np.minimum(

    per_severity[
        "p_value"
    ]
    * len(
        per_severity
    ),

    1.0,
)


per_severity = (

    per_severity

    .sort_values(
        [
            "shift_type",
            "shift_value",
        ]
    )

    .reset_index(
        drop=True
    )
)


per_severity.to_csv(
    PER_SEVERITY_FILE,
    index=False,
)


print(
    per_severity[
        [
            "shift_type",
            "shift_value",
            "f_stat",
            "p_value",
            "p_fdr_bh",
            "reject_fdr_05",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SIMPLE DESCRIPTIVE EFFECT PATTERNS
#
# These are descriptive only.
# They do not replace H6.5/H6.6.
# ============================================================

print()
print("=" * 78)
print("DESCRIPTIVE TRAJECTORY EFFECTS")
print("=" * 78)


for shift_type in SHIFT_FAMILIES:

    df = (

        delta_auc.loc[
            delta_auc["shift_type"]
            == shift_type
        ]
    )


    mean_effect = (
        df[
            "delta_auc_mse"
        ].mean()
    )


    median_effect = (
        df[
            "delta_auc_mse"
        ].median()
    )


    fraction_improved = (
        (
            df[
                "delta_auc_mse"
            ]
            < 0
        )
        .mean()
    )


    print()

    print(
        f"{shift_type}:"
    )

    print(
        f"  mean delta AUC MSE: "
        f"{mean_effect:.6f}"
    )

    print(
        f"  median delta AUC MSE: "
        f"{median_effect:.6f}"
    )

    print(
        f"  fraction seed-level "
        f"effects < 0: "
        f"{fraction_improved:.6f}"
    )


# ============================================================
# WRITE ANALYSIS CONFIG
# ============================================================

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        ANALYSIS_CONFIG,
        f,
        indent=2,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_lines = []

report_lines.append(
    "PHASE 6 G2 CONFIRMATION ANALYSIS"
)

report_lines.append(
    "=" * 60
)

report_lines.append("")

report_lines.append(
    f"Raw confirmation rows: {len(raw)}"
)

report_lines.append(
    f"Unique trained models: {raw['run_id'].nunique()}"
)

report_lines.append(
    f"Trajectory AUC rows: {len(auc)}"
)

report_lines.append(
    f"Paired delta-AUC rows: {len(delta_auc)}"
)

report_lines.append(
    f"Pointwise paired delta-MSE rows: {len(pointwise)}"
)

report_lines.append("")

report_lines.append(
    "H6.5: trajectory-level D x A x M"
)

report_lines.append(
    h65.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "H6.6: D x A x M x severity"
)

report_lines.append(
    h66.to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "Secondary per-severity D x A x M"
)

report_lines.append(
    per_severity[
        [
            "shift_type",
            "shift_value",
            "f_stat",
            "p_value",
            "p_fdr_bh",
            "reject_fdr_05",
        ]
    ].to_string(
        index=False
    )
)

report_lines.append("")

report_lines.append(
    "INTERPRETATION RULE"
)

report_lines.append(
    (
        "This analysis must be interpreted as an "
        "independent G2 confirmation. Results must not "
        "be used to alter G2, augmentation definitions, "
        "model configurations, seeds, or severity grids."
    )
)

report_lines.append("")

report_lines.append(
    (
        "Cross-generator replication conclusions are "
        "deferred to the separate Phase 5 versus Phase 6 "
        "comparison analysis."
    )
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 78)
print("PHASE 6 ANALYSIS COMPUTATIONAL CHECK")
print("=" * 78)

print(
    f"Raw records:              {len(raw):>6}"
)

print(
    f"Trajectory AUC rows:      {len(auc):>6}"
)

print(
    f"Paired delta-AUC rows:    {len(delta_auc):>6}"
)

print(
    f"Pointwise delta rows:     {len(pointwise):>6}"
)

print(
    f"H6.5 tests:               {len(h65):>6}"
)

print(
    f"H6.6 tests:               {len(h66):>6}"
)

print(
    f"Per-severity tests:       {len(per_severity):>6}"
)


# ============================================================
# FINAL HARD ASSERTIONS
# ============================================================

assert len(raw) == 30000
assert raw["run_id"].nunique() == 1500

assert len(auc) == 3000
assert len(delta_auc) == 2400

assert len(pointwise) == 15600

assert len(
    pointwise[
        pointwise["shift_type"]
        == "temperature"
    ]
) == 7200

assert len(
    pointwise[
        pointwise["shift_type"]
        == "humidity"
    ]
) == 8400

assert len(h65) == 2
assert len(h66) == 2

assert len(per_severity) == 13

assert np.isfinite(
    delta_auc[
        "delta_auc_mse"
    ]
).all()

assert np.isfinite(
    pointwise[
        "delta_mse"
    ]
).all()

assert np.isfinite(
    h65[
        [
            "f_stat",
            "p_value",
        ]
    ].to_numpy(
        dtype=float
    )
).all()

assert np.isfinite(
    h66[
        [
            "f_stat",
            "p_value",
        ]
    ].to_numpy(
        dtype=float
    )
).all()


print()
print(
    "PHASE 6 CONFIRMATION ANALYSIS: "
    "COMPUTATIONAL PASS"
)

print()

print(
    f"Results written to:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "No Phase 5 vs Phase 6 replication "
    "claim has been made by this script."
)