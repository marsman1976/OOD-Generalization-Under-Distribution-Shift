from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 4 CONFIRMATION — DESCRIPTIVE ANALYSIS
#
# Purpose:
#
# 1. Load the frozen Phase 4 confirmation results.
# 2. Re-check structural integrity.
# 3. Construct paired augmentation effects:
#
#       delta_mse = MSE(A) - MSE(A0)
#
# 4. Describe augmentation effects by:
#
#       A
#       M
#       shift family
#       severity
#
# 5. Calculate:
#
#       mean delta MSE
#       median delta MSE
#       SD
#       fraction improved
#       fraction harmed
#       severity trajectories
#       trajectory AUC
#       delta AUC
#
# IMPORTANT:
#
# This script is DESCRIPTIVE.
#
# It does NOT perform the preregistered formal
# A × M × V hypothesis test.
#
# Formal inference will be performed separately.
# ============================================================


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "experiment_4"
    / "confirmation"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "master_results.csv"
)

CONFIG_FILE = (
    RESULTS_DIR
    / "confirmation_config.json"
)

ANALYSIS_DIR = (
    RESULTS_DIR
    / "analysis"
)

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

PAIRED_FILE = (
    ANALYSIS_DIR
    / "paired_augmentation_effects.csv"
)

SUMMARY_FILE = (
    ANALYSIS_DIR
    / "augmentation_model_summary.csv"
)

SEVERITY_SUMMARY_FILE = (
    ANALYSIS_DIR
    / "severity_summary.csv"
)

AUC_FILE = (
    ANALYSIS_DIR
    / "trajectory_auc.csv"
)

DELTA_AUC_FILE = (
    ANALYSIS_DIR
    / "delta_auc.csv"
)

RAW_AUC_FILE = (
    ANALYSIS_DIR
    / "raw_mse_auc.csv"
)

REPORT_FILE = (
    ANALYSIS_DIR
    / "descriptive_report.json"
)

TEMP_PLOT_FILE = (
    ANALYSIS_DIR
    / "temperature_delta_trajectories.png"
)

HUMIDITY_PLOT_FILE = (
    ANALYSIS_DIR
    / "humidity_delta_trajectories.png"
)


# ============================================================
# Frozen Phase 4 factors
# ============================================================

EXPECTED_DATASET = "D00"

AUGMENTATIONS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]

NON_BASELINE_AUGMENTATIONS = [
    "A1",
    "A2",
    "A3",
    "A4",
]

MODELS = [
    "M0",
    "M1",
    "M2",
]

SEEDS = [
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    111,
]

TEMPERATURE_LEVELS = [
    23,
    25,
    27,
    29,
    31,
    33,
]

HUMIDITY_LEVELS = [
    65,
    70,
    75,
    80,
    85,
    90,
    92,
]


# ============================================================
# Utility functions
# ============================================================

def fraction_negative(series):

    return float(
        np.mean(
            np.asarray(series)
            < 0
        )
    )


def fraction_positive(series):

    return float(
        np.mean(
            np.asarray(series)
            > 0
        )
    )


def fraction_zero(series):

    return float(
        np.mean(
            np.isclose(
                np.asarray(series),
                0.0,
            )
        )
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

    # np.trapezoid exists in newer NumPy.
    # np.trapz keeps compatibility with older versions.
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


# ============================================================
# Load confirmation data
# ============================================================

def load_data():

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            RESULTS_FILE
        )

    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            CONFIG_FILE
        )

    df = pd.read_csv(
        RESULTS_FILE
    )

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    return df, config


# ============================================================
# Structural integrity
# ============================================================

def check_confirmation_integrity(
    df,
    config,
):

    print()
    print("=" * 76)
    print("CONFIRMATION INTEGRITY")
    print("=" * 76)

    print(
        "Records:",
        len(df)
    )

    print(
        "Unique models:",
        df["run_id"].nunique()
    )

    assert len(df) == 3000

    assert (
        df["run_id"].nunique()
        == 150
    )

    assert set(
        df["dataset_id"].unique()
    ) == {
        EXPECTED_DATASET
    }

    assert set(
        df["augmentation_id"].unique()
    ) == set(
        AUGMENTATIONS
    )

    assert set(
        df["model_config"].unique()
    ) == set(
        MODELS
    )

    assert set(
        df["seed"].unique()
    ) == set(
        SEEDS
    )

    duplicates = (
        df
        .duplicated(
            subset=[
                "run_id",
                "experiment",
                "environment",
            ]
        )
        .sum()
    )

    print(
        "Duplicates:",
        duplicates
    )

    assert duplicates == 0

    run_counts = (
        df
        .groupby(
            "run_id"
        )
        .size()
    )

    assert (
        run_counts == 20
    ).all()

    for column in [
        "best_validation_loss",
        "mse",
        "mae",
    ]:

        assert (
            df[column]
            .notna()
            .all()
        )

        assert np.isfinite(
            df[column]
        ).all()

    experiment_counts = (
        df["experiment"]
        .value_counts()
    )

    assert (
        experiment_counts["EXP1"]
        == 600
    )

    assert (
        experiment_counts["EXP1B"]
        == 450
    )

    assert (
        experiment_counts["EXP1C"]
        == 1950
    )

    assert (
        config["n_models"]
        == 150
    )

    assert (
        config["expected_records"]
        == 3000
    )

    assert (
        config["primary_interaction"]
        == "A x M x V"
    )

    print(
        "Confirmation integrity: PASS"
    )


# ============================================================
# Construct paired augmentation effects
#
# Baseline:
#
# A0 with same:
#
# dataset
# model
# seed
# experiment
# environment
# shift type
# severity
# shift value
#
# delta_mse = mse_augmented - mse_A0
#
# Negative = augmentation improved performance.
# ============================================================

def build_paired_dataset(df):

    print()
    print("=" * 76)
    print("BUILDING PAIRED AUGMENTATION DATASET")
    print("=" * 76)

    pair_keys = [
        "dataset_id",
        "model_config",
        "seed",
        "experiment",
        "environment",
        "shift_type",
        "severity",
        "shift_value",
    ]

    baseline = (
        df[
            df["augmentation_id"]
            == "A0"
        ]
        .copy()
    )

    augmented = (
        df[
            df["augmentation_id"]
            != "A0"
        ]
        .copy()
    )

    # Keep baseline metrics separately.
    baseline = baseline[
        pair_keys
        + [
            "mse",
            "mae",
            "best_validation_loss",
        ]
    ].rename(
        columns={
            "mse":
                "mse_A0",

            "mae":
                "mae_A0",

            "best_validation_loss":
                "validation_loss_A0",
        }
    )

    paired = augmented.merge(
        baseline,
        on=pair_keys,
        how="left",
        validate="many_to_one",
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    assert (
        paired["mse_A0"]
        .notna()
        .all()
    )

    assert (
        paired["mae_A0"]
        .notna()
        .all()
    )

    # --------------------------------------------------------
    # Paired estimands
    # --------------------------------------------------------

    paired[
        "delta_mse"
    ] = (

        paired["mse"]

        - paired["mse_A0"]

    )

    paired[
        "delta_mae"
    ] = (

        paired["mae"]

        - paired["mae_A0"]

    )

    paired[
        "improved_mse"
    ] = (

        paired["delta_mse"]
        < 0

    )

    paired[
        "harmed_mse"
    ] = (

        paired["delta_mse"]
        > 0

    )

    # --------------------------------------------------------
    # Expected total paired records
    #
    # 4 non-A0 augmentations
    # × 3 models
    # × 10 seeds
    # × 20 environments
    #
    # = 2400
    # --------------------------------------------------------

    assert (
        len(paired)
        == 2400
    )

    exp1c = paired[
        paired["experiment"]
        == "EXP1C"
    ]

    temperature = exp1c[
        exp1c["shift_type"]
        == "temperature"
    ]

    humidity = exp1c[
        exp1c["shift_type"]
        == "humidity"
    ]

    assert (
        len(exp1c)
        == 1560
    )

    assert (
        len(temperature)
        == 720
    )

    assert (
        len(humidity)
        == 840
    )

    print(
        "All paired records:",
        len(paired)
    )

    print(
        "EXP1C paired records:",
        len(exp1c)
    )

    print(
        "Temperature:",
        len(temperature)
    )

    print(
        "Humidity:",
        len(humidity)
    )

    print(
        "Paired dataset integrity: PASS"
    )

    return paired


# ============================================================
# Overall A × M descriptive summary
# ============================================================

def create_augmentation_model_summary(
    paired
):

    summary = (

        paired

        .groupby(
            [
                "augmentation_id",
                "model_config",
            ]
        )

        .agg(

            n=(
                "delta_mse",
                "size",
            ),

            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),

            median_delta_mse=(
                "delta_mse",
                "median",
            ),

            sd_delta_mse=(
                "delta_mse",
                "std",
            ),

            mean_delta_mae=(
                "delta_mae",
                "mean",
            ),

            median_delta_mae=(
                "delta_mae",
                "median",
            ),

            mean_raw_mse=(
                "mse",
                "mean",
            ),

            mean_A0_mse=(
                "mse_A0",
                "mean",
            ),

        )

        .reset_index()

    )

    fractions = (

        paired

        .groupby(
            [
                "augmentation_id",
                "model_config",
            ]
        )["delta_mse"]

        .agg(

            fraction_improved=
                fraction_negative,

            fraction_harmed=
                fraction_positive,

            fraction_equal=
                fraction_zero,

        )

        .reset_index()

    )

    summary = summary.merge(

        fractions,

        on=[
            "augmentation_id",
            "model_config",
        ],

        how="left",

        validate="one_to_one",

    )

    assert (
        len(summary)
        == 12
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    return summary


# ============================================================
# Severity-specific descriptive summary
# ============================================================

def create_severity_summary(
    paired
):

    exp1c = paired[
        paired["experiment"]
        == "EXP1C"
    ].copy()

    summary = (

        exp1c

        .groupby(
            [
                "shift_type",
                "severity",
                "shift_value",
                "augmentation_id",
                "model_config",
            ],
            dropna=False,
        )

        .agg(

            n=(
                "delta_mse",
                "size",
            ),

            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),

            median_delta_mse=(
                "delta_mse",
                "median",
            ),

            sd_delta_mse=(
                "delta_mse",
                "std",
            ),

            mean_delta_mae=(
                "delta_mae",
                "mean",
            ),

            mean_raw_mse=(
                "mse",
                "mean",
            ),

            mean_A0_mse=(
                "mse_A0",
                "mean",
            ),

        )

        .reset_index()

    )

    fractions = (

        exp1c

        .groupby(
            [
                "shift_type",
                "severity",
                "shift_value",
                "augmentation_id",
                "model_config",
            ],
            dropna=False,
        )["delta_mse"]

        .agg(

            fraction_improved=
                fraction_negative,

            fraction_harmed=
                fraction_positive,

        )

        .reset_index()

    )

    summary = summary.merge(

        fractions,

        on=[
            "shift_type",
            "severity",
            "shift_value",
            "augmentation_id",
            "model_config",
        ],

        how="left",

        validate="one_to_one",

    )

    # Temperature:
    # 6 severity × 4 augmentation × 3 models = 72
    #
    # Humidity:
    # 7 × 4 × 3 = 84
    #
    # Total = 156

    assert (
        len(summary)
        == 156
    )

    assert (
        summary["n"]
        == 10
    ).all()

    summary.to_csv(
        SEVERITY_SUMMARY_FILE,
        index=False,
    )

    return summary


# ============================================================
# Raw MSE trajectory AUC
#
# This is calculated at the seed level.
#
# One AUC per:
#
# shift family × A × M × seed
# ============================================================

def calculate_raw_auc(df):

    exp1c = df[
        df["experiment"]
        == "EXP1C"
    ].copy()

    records = []

    group_columns = [
        "shift_type",
        "augmentation_id",
        "model_config",
        "seed",
    ]

    for keys, group in exp1c.groupby(
        group_columns
    ):

        (
            shift_type,
            augmentation_id,
            model_config,
            seed,
        ) = keys

        group = group.sort_values(
            "shift_value"
        )

        expected_n = (

            6
            if shift_type
            == "temperature"

            else 7

        )

        assert (
            len(group)
            == expected_n
        )

        auc = trapezoid_auc(

            group[
                "shift_value"
            ],

            group[
                "mse"
            ],

        )

        records.append({

            "shift_type":
                shift_type,

            "augmentation_id":
                augmentation_id,

            "model_config":
                model_config,

            "seed":
                seed,

            "raw_mse_auc":
                auc,

        })

    auc_df = pd.DataFrame(
        records
    )

    # 2 shift families
    # × 5 augmentations
    # × 3 models
    # × 10 seeds
    #
    # = 300

    assert (
        len(auc_df)
        == 300
    )

    auc_df.to_csv(
        RAW_AUC_FILE,
        index=False,
    )

    return auc_df


# ============================================================
# Paired delta trajectory AUC
#
# One AUC per:
#
# shift family × non-A0 augmentation × model × seed
#
# Negative delta AUC means augmentation reduced error
# across the trajectory relative to A0.
# ============================================================

def calculate_delta_auc(
    paired
):

    exp1c = paired[
        paired["experiment"]
        == "EXP1C"
    ].copy()

    records = []

    group_columns = [
        "shift_type",
        "augmentation_id",
        "model_config",
        "seed",
    ]

    for keys, group in exp1c.groupby(
        group_columns
    ):

        (
            shift_type,
            augmentation_id,
            model_config,
            seed,
        ) = keys

        group = group.sort_values(
            "shift_value"
        )

        expected_n = (

            6
            if shift_type
            == "temperature"

            else 7

        )

        assert (
            len(group)
            == expected_n
        )

        delta_auc = trapezoid_auc(

            group[
                "shift_value"
            ],

            group[
                "delta_mse"
            ],

        )

        records.append({

            "shift_type":
                shift_type,

            "augmentation_id":
                augmentation_id,

            "model_config":
                model_config,

            "seed":
                seed,

            "delta_auc":
                delta_auc,

        })

    delta_auc_df = pd.DataFrame(
        records
    )

    # 2 shift families
    # × 4 augmentations
    # × 3 models
    # × 10 seeds
    #
    # = 240

    assert (
        len(delta_auc_df)
        == 240
    )

    delta_auc_df.to_csv(
        AUC_FILE,
        index=False,
    )

    return delta_auc_df


# ============================================================
# Delta-AUC descriptive summary
# ============================================================

def summarize_delta_auc(
    delta_auc_df
):

    summary = (

        delta_auc_df

        .groupby(
            [
                "shift_type",
                "augmentation_id",
                "model_config",
            ]
        )

        .agg(

            n=(
                "delta_auc",
                "size",
            ),

            mean_delta_auc=(
                "delta_auc",
                "mean",
            ),

            median_delta_auc=(
                "delta_auc",
                "median",
            ),

            sd_delta_auc=(
                "delta_auc",
                "std",
            ),

        )

        .reset_index()

    )

    fractions = (

        delta_auc_df

        .groupby(
            [
                "shift_type",
                "augmentation_id",
                "model_config",
            ]
        )["delta_auc"]

        .agg(

            fraction_lower_auc=
                fraction_negative,

            fraction_higher_auc=
                fraction_positive,

        )

        .reset_index()

    )

    summary = summary.merge(

        fractions,

        on=[
            "shift_type",
            "augmentation_id",
            "model_config",
        ],

        how="left",

        validate="one_to_one",

    )

    # 2 shift types
    # × 4 augmentation
    # × 3 model
    #
    # = 24

    assert (
        len(summary)
        == 24
    )

    assert (
        summary["n"]
        == 10
    ).all()

    summary.to_csv(
        DELTA_AUC_FILE,
        index=False,
    )

    return summary


# ============================================================
# Plot severity trajectories
#
# One figure for temperature.
# One figure for humidity.
#
# To keep plots readable:
#
# each line = augmentation × model
#
# y = mean paired delta MSE
#
# Negative values = augmentation helps.
# ============================================================

def plot_shift_family(
    severity_summary,
    shift_type,
    output_file,
):

    data = severity_summary[
        severity_summary[
            "shift_type"
        ]
        == shift_type
    ].copy()

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for augmentation_id in NON_BASELINE_AUGMENTATIONS:

        for model_config in MODELS:

            subset = data[
                (
                    data[
                        "augmentation_id"
                    ]
                    == augmentation_id
                )
                &
                (
                    data[
                        "model_config"
                    ]
                    == model_config
                )
            ].sort_values(
                "shift_value"
            )

            ax.plot(
                subset[
                    "shift_value"
                ],
                subset[
                    "mean_delta_mse"
                ],
                marker="o",
                label=(
                    f"{augmentation_id}-"
                    f"{model_config}"
                ),
            )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_xlabel(
        "Shift value"
    )

    ax.set_ylabel(
        "Mean paired ΔMSE"
    )

    ax.set_title(
        (
            "Phase 4 confirmation: "
            f"{shift_type} shift"
        )
    )

    ax.legend(
        fontsize=8,
        ncol=3,
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=180,
    )

    plt.close(
        fig
    )


# ============================================================
# Console report
# ============================================================

def print_descriptive_results(
    paired,
    summary,
    delta_auc_summary,
):

    print()
    print("=" * 76)
    print("PHASE 4 DESCRIPTIVE RESULTS")
    print("=" * 76)

    print()
    print(
        "Overall paired augmentation effect:"
    )

    print(
        "Mean delta MSE:",
        paired["delta_mse"].mean()
    )

    print(
        "Median delta MSE:",
        paired["delta_mse"].median()
    )

    print(
        "Fraction improved:",
        fraction_negative(
            paired["delta_mse"]
        )
    )

    print(
        "Fraction harmed:",
        fraction_positive(
            paired["delta_mse"]
        )
    )

    print()
    print(
        "A x M summary:"
    )

    display_columns = [
        "augmentation_id",
        "model_config",
        "n",
        "mean_delta_mse",
        "median_delta_mse",
        "fraction_improved",
        "fraction_harmed",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Delta-AUC summary:"
    )

    auc_columns = [
        "shift_type",
        "augmentation_id",
        "model_config",
        "mean_delta_auc",
        "median_delta_auc",
        "fraction_lower_auc",
    ]

    print(
        delta_auc_summary[
            auc_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# Save compact JSON report
# ============================================================

def save_report(
    paired,
    severity_summary,
    delta_auc_summary,
):

    exp1c = paired[
        paired["experiment"]
        == "EXP1C"
    ]

    temperature = exp1c[
        exp1c["shift_type"]
        == "temperature"
    ]

    humidity = exp1c[
        exp1c["shift_type"]
        == "humidity"
    ]

    report = {

        "phase":
            "PHASE4",

        "analysis_type":
            "DESCRIPTIVE_CONFIRMATION",

        "paired_estimand":
            (
                "delta_mse = "
                "MSE(A) - MSE(A0)"
            ),

        "interpretation":
            (
                "negative delta_mse = "
                "augmentation improvement"
            ),

        "n_confirmation_records":
            3000,

        "n_paired_records":
            int(
                len(paired)
            ),

        "n_exp1c_paired":
            int(
                len(exp1c)
            ),

        "n_temperature_pairs":
            int(
                len(temperature)
            ),

        "n_humidity_pairs":
            int(
                len(humidity)
            ),

        "overall_mean_delta_mse":
            float(
                paired[
                    "delta_mse"
                ].mean()
            ),

        "overall_median_delta_mse":
            float(
                paired[
                    "delta_mse"
                ].median()
            ),

        "overall_fraction_improved":
            fraction_negative(
                paired[
                    "delta_mse"
                ]
            ),

        "overall_fraction_harmed":
            fraction_positive(
                paired[
                    "delta_mse"
                ]
            ),

        "n_severity_summary_cells":
            int(
                len(
                    severity_summary
                )
            ),

        "n_delta_auc_summary_cells":
            int(
                len(
                    delta_auc_summary
                )
            ),

        "formal_inference_performed":
            False,

        "next_analysis":
            (
                "preregistered "
                "A x M x V inference"
            ),

    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 76
    )

    print(
        "PHASE 4 CONFIRMATION "
        "DESCRIPTIVE ANALYSIS"
    )

    print(
        "=" * 76
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df, config = load_data()

    # --------------------------------------------------------
    # Confirmation integrity
    # --------------------------------------------------------

    check_confirmation_integrity(
        df,
        config,
    )

    # --------------------------------------------------------
    # Pair A1-A4 against A0
    # --------------------------------------------------------

    paired = build_paired_dataset(
        df
    )

    paired.to_csv(
        PAIRED_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Overall A × M summary
    # --------------------------------------------------------

    summary = (
        create_augmentation_model_summary(
            paired
        )
    )

    # --------------------------------------------------------
    # Severity trajectories
    # --------------------------------------------------------

    severity_summary = (
        create_severity_summary(
            paired
        )
    )

    # --------------------------------------------------------
    # Raw trajectory AUC
    # --------------------------------------------------------

    calculate_raw_auc(
        df
    )

    # --------------------------------------------------------
    # Paired delta trajectory AUC
    # --------------------------------------------------------

    delta_auc_df = (
        calculate_delta_auc(
            paired
        )
    )

    delta_auc_summary = (
        summarize_delta_auc(
            delta_auc_df
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_shift_family(
        severity_summary,
        "temperature",
        TEMP_PLOT_FILE,
    )

    plot_shift_family(
        severity_summary,
        "humidity",
        HUMIDITY_PLOT_FILE,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_descriptive_results(
        paired,
        summary,
        delta_auc_summary,
    )

    save_report(
        paired,
        severity_summary,
        delta_auc_summary,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "PHASE 4 DESCRIPTIVE "
        "ANALYSIS: PASS"
    )
    print("=" * 76)

    print()
    print(
        "Outputs saved to:"
    )

    print(
        ANALYSIS_DIR
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are descriptive results only."
    )

    print(
        "Do not interpret them as formal "
        "hypothesis-test results."
    )


if __name__ == "__main__":

    main()