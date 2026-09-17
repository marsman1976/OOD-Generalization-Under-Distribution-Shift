from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "experiment_2"
    / "confirmation"
    / "master_results.csv"
)

OUTPUT = (
    ROOT
    / "results"
    / "experiment_2"
    / "confirmation"
    / "analysis"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CONFIRMATION DATA
# ============================================================

print("=" * 75)
print("PHASE 2 CONFIRMATION ANALYSIS")
print("=" * 75)

df = pd.read_csv(INPUT)

print("\nInput:", INPUT)
print("Records:", len(df))
print("Models:", df["model_id"].nunique())

print(
    "Datasets:",
    sorted(df["dataset_id"].unique())
)

print(
    "Augmentations:",
    sorted(df["augmentation_id"].unique())
)

print(
    "Seeds:",
    sorted(df["seed"].unique())
)


# ============================================================
# STRICT INTEGRITY CHECK
# ============================================================

EXPECTED_DATASETS = {
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
}

EXPECTED_AUGMENTATIONS = {
    "A0", "A1", "A2", "A3", "A4",
}

EXPECTED_SEEDS = {
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
}

assert len(df) == 10000
assert df["model_id"].nunique() == 500

assert set(df["dataset_id"].unique()) == EXPECTED_DATASETS

assert (
    set(df["augmentation_id"].unique())
    == EXPECTED_AUGMENTATIONS
)

assert set(df["seed"].unique()) == EXPECTED_SEEDS


duplicate_count = df.duplicated(
    subset=[
        "model_id",
        "experiment",
        "environment",
    ]
).sum()

assert duplicate_count == 0


model_counts = (
    df.groupby("model_id")
    .size()
)

assert (model_counts == 20).all()


experiment_counts = (
    df["experiment"]
    .value_counts()
)

assert experiment_counts["EXP1"] == 2000
assert experiment_counts["EXP1B"] == 1500
assert experiment_counts["EXP1C"] == 6500


print("\nCONFIRMATION INPUT INTEGRITY: PASS")


# ============================================================
# BASIC SUMMARY
# ============================================================

basic_summary = (
    df.groupby([
        "dataset_id",
        "augmentation_id",
        "experiment",
        "environment",
    ])
    .agg(
        mean_mse=("mse", "mean"),
        std_mse=("mse", "std"),
        median_mse=("mse", "median"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        n=("mse", "size"),
    )
    .reset_index()
)

basic_summary.to_csv(
    OUTPUT / "environment_summary.csv",
    index=False,
)


# ============================================================
# PAIRED AUGMENTATION EFFECT
#
# delta MSE =
# MSE(Ax) - MSE(A0)
#
# negative = augmentation improved performance
# positive = augmentation harmed performance
# ============================================================

print("\nCalculating paired augmentation effects...")


baseline = (
    df[
        df["augmentation_id"] == "A0"
    ][
        [
            "dataset_id",
            "seed",
            "experiment",
            "environment",
            "shift_type",
            "severity",
            "shift_value",
            "mse",
            "mae",
        ]
    ]
    .rename(
        columns={
            "mse": "mse_A0",
            "mae": "mae_A0",
        }
    )
)


augmented = df[
    df["augmentation_id"] != "A0"
].copy()


paired = augmented.merge(
    baseline,
    on=[
        "dataset_id",
        "seed",
        "experiment",
        "environment",
    ],
    how="left",
    suffixes=("", "_baseline"),
)


assert paired["mse_A0"].notna().all()


paired["delta_mse"] = (
    paired["mse"]
    - paired["mse_A0"]
)

paired["delta_mae"] = (
    paired["mae"]
    - paired["mae_A0"]
)

paired["improved_mse"] = (
    paired["delta_mse"] < 0
)

paired["improved_mae"] = (
    paired["delta_mae"] < 0
)


paired.to_csv(
    OUTPUT / "paired_results.csv",
    index=False,
)


# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def mean_ci(values, confidence=0.95):

    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            n,
        )

    mean = np.mean(values)

    se = stats.sem(values)

    ci = stats.t.interval(
        confidence,
        df=n - 1,
        loc=mean,
        scale=se,
    )

    return (
        mean,
        ci[0],
        ci[1],
        n,
    )


# ============================================================
# PAIRED SUMMARY BY D × A
# ============================================================

rows = []

for (
    dataset_id,
    augmentation_id
), group in paired.groupby(
    [
        "dataset_id",
        "augmentation_id",
    ]
):

    mean_delta, ci_low, ci_high, n = (
        mean_ci(group["delta_mse"])
    )

    rows.append({
        "dataset_id": dataset_id,
        "augmentation_id":
            augmentation_id,

        "n": n,

        "mean_delta_mse":
            mean_delta,

        "median_delta_mse":
            group["delta_mse"].median(),

        "std_delta_mse":
            group["delta_mse"].std(),

        "ci95_low":
            ci_low,

        "ci95_high":
            ci_high,

        "fraction_improved":
            group["improved_mse"].mean(),

        "mean_delta_mae":
            group["delta_mae"].mean(),
    })


paired_summary = pd.DataFrame(rows)

paired_summary.to_csv(
    OUTPUT / "paired_delta_summary.csv",
    index=False,
)


print(
    "Paired D x A rows:",
    len(paired_summary)
)

# 10 D × 4 non-baseline augmentations
assert len(paired_summary) == 40


# ============================================================
# SUMMARY BY EXPERIMENT
# ============================================================

experiment_delta_summary = (
    paired.groupby([
        "dataset_id",
        "augmentation_id",
        "experiment",
    ])
    .agg(
        mean_delta_mse=(
            "delta_mse",
            "mean"
        ),
        median_delta_mse=(
            "delta_mse",
            "median"
        ),
        std_delta_mse=(
            "delta_mse",
            "std"
        ),
        fraction_improved=(
            "improved_mse",
            "mean"
        ),
        n=(
            "delta_mse",
            "size"
        ),
    )
    .reset_index()
)

experiment_delta_summary.to_csv(
    OUTPUT /
    "experiment_delta_summary.csv",
    index=False,
)


# ============================================================
# SEVERITY ANALYSIS
# ============================================================

severity_df = df[
    df["experiment"] == "EXP1C"
].copy()


severity_summary = (
    severity_df.groupby([
        "shift_type",
        "severity",
        "shift_value",
        "dataset_id",
        "augmentation_id",
    ])
    .agg(
        mean_mse=("mse", "mean"),
        std_mse=("mse", "std"),
        median_mse=("mse", "median"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        n=("mse", "size"),
    )
    .reset_index()
)


severity_summary.to_csv(
    OUTPUT / "severity_summary.csv",
    index=False,
)


print(
    "Severity summary rows:",
    len(severity_summary)
)

# 10D × 5A × (6 temp + 7 humidity)
assert len(severity_summary) == 650


# ============================================================
# PAIRED DELTA BY SEVERITY
# ============================================================

severity_paired = paired[
    paired["experiment"] == "EXP1C"
].copy()


severity_delta_summary = (
    severity_paired.groupby([
        "shift_type",
        "severity",
        "shift_value",
        "dataset_id",
        "augmentation_id",
    ])
    .agg(
        mean_delta_mse=(
            "delta_mse",
            "mean"
        ),
        std_delta_mse=(
            "delta_mse",
            "std"
        ),
        median_delta_mse=(
            "delta_mse",
            "median"
        ),
        fraction_improved=(
            "improved_mse",
            "mean"
        ),
        n=(
            "delta_mse",
            "size"
        ),
    )
    .reset_index()
)


severity_delta_summary.to_csv(
    OUTPUT /
    "severity_delta_summary.csv",
    index=False,
)


# ============================================================
# DEGRADATION FROM S0
# ============================================================

print("Calculating degradation curves...")


degradation_records = []


for (
    shift_type,
    dataset_id,
    augmentation_id,
    seed
), group in severity_df.groupby(
    [
        "shift_type",
        "dataset_id",
        "augmentation_id",
        "seed",
    ]
):

    group = group.sort_values(
        "severity"
    )

    baseline_row = group[
        group["severity"] == 0
    ]

    if baseline_row.empty:
        continue

    mse_s0 = baseline_row[
        "mse"
    ].iloc[0]

    mae_s0 = baseline_row[
        "mae"
    ].iloc[0]


    for _, row in group.iterrows():

        degradation_records.append({
            "shift_type":
                shift_type,

            "dataset_id":
                dataset_id,

            "augmentation_id":
                augmentation_id,

            "seed":
                seed,

            "severity":
                row["severity"],

            "shift_value":
                row["shift_value"],

            "mse":
                row["mse"],

            "mse_s0":
                mse_s0,

            "delta_from_s0":
                row["mse"]
                - mse_s0,

            "mae":
                row["mae"],

            "delta_mae_from_s0":
                row["mae"]
                - mae_s0,
        })


degradation = pd.DataFrame(
    degradation_records
)


degradation.to_csv(
    OUTPUT / "degradation.csv",
    index=False,
)


# ============================================================
# ROBUSTNESS AUC PER MODEL
#
# IMPORTANT:
# use actual shift_value as x axis.
#
# Temperature and humidity AUC should be interpreted
# separately because their x-axis units differ.
# ============================================================

print("Calculating robustness AUC...")


auc_records = []


for (
    shift_type,
    dataset_id,
    augmentation_id,
    seed
), group in severity_df.groupby(
    [
        "shift_type",
        "dataset_id",
        "augmentation_id",
        "seed",
    ]
):

    group = group.sort_values(
        "shift_value"
    )

    x = group[
        "shift_value"
    ].to_numpy(
        dtype=float
    )

    y = group[
        "mse"
    ].to_numpy(
        dtype=float
    )

    auc = np.trapezoid(
        y,
        x,
    )

    width = (
        x.max()
        - x.min()
    )

    normalized_auc = (
        auc / width
        if width > 0
        else np.nan
    )


    auc_records.append({
        "shift_type":
            shift_type,

        "dataset_id":
            dataset_id,

        "augmentation_id":
            augmentation_id,

        "seed":
            seed,

        "auc_mse":
            auc,

        "normalized_auc_mse":
            normalized_auc,
    })


auc_per_model = pd.DataFrame(
    auc_records
)


auc_per_model.to_csv(
    OUTPUT /
    "robustness_auc_per_model.csv",
    index=False,
)


# ============================================================
# AUC SUMMARY
# ============================================================

auc_summary = (
    auc_per_model.groupby([
        "shift_type",
        "dataset_id",
        "augmentation_id",
    ])
    .agg(
        mean_auc=(
            "auc_mse",
            "mean"
        ),
        std_auc=(
            "auc_mse",
            "std"
        ),
        median_auc=(
            "auc_mse",
            "median"
        ),
        mean_normalized_auc=(
            "normalized_auc_mse",
            "mean"
        ),
        std_normalized_auc=(
            "normalized_auc_mse",
            "std"
        ),
        n=(
            "auc_mse",
            "size"
        ),
    )
    .reset_index()
)


auc_summary.to_csv(
    OUTPUT /
    "robustness_auc_summary.csv",
    index=False,
)


print(
    "AUC summary rows:",
    len(auc_summary)
)

# 2 shift types × 10D × 5A
assert len(auc_summary) == 100


# ============================================================
# AUC PAIRED EFFECT RELATIVE TO A0
# ============================================================

auc_baseline = (
    auc_per_model[
        auc_per_model[
            "augmentation_id"
        ] == "A0"
    ][
        [
            "shift_type",
            "dataset_id",
            "seed",
            "auc_mse",
            "normalized_auc_mse",
        ]
    ]
    .rename(
        columns={
            "auc_mse":
                "auc_A0",

            "normalized_auc_mse":
                "normalized_auc_A0",
        }
    )
)


auc_delta = (
    auc_per_model[
        auc_per_model[
            "augmentation_id"
        ] != "A0"
    ]
    .merge(
        auc_baseline,
        on=[
            "shift_type",
            "dataset_id",
            "seed",
        ],
        how="left",
    )
)


auc_delta[
    "delta_auc"
] = (
    auc_delta["auc_mse"]
    - auc_delta["auc_A0"]
)


auc_delta[
    "delta_normalized_auc"
] = (
    auc_delta[
        "normalized_auc_mse"
    ]
    - auc_delta[
        "normalized_auc_A0"
    ]
)


auc_delta.to_csv(
    OUTPUT /
    "robustness_auc_delta.csv",
    index=False,
)


# ============================================================
# PAIRED T TESTS
#
# Exploratory / secondary:
# compare augmentation against A0 within D and environment.
#
# Do not treat these alone as the formal D×A interaction test.
# ============================================================

print("Running paired augmentation tests...")


test_records = []


for (
    dataset_id,
    augmentation_id,
    experiment,
    environment
), group in paired.groupby(
    [
        "dataset_id",
        "augmentation_id",
        "experiment",
        "environment",
    ]
):

    values = group[
        "delta_mse"
    ].dropna().to_numpy()

    if len(values) < 2:
        continue

    t_stat, p_value = (
        stats.ttest_1samp(
            values,
            popmean=0.0,
        )
    )

    test_records.append({
        "dataset_id":
            dataset_id,

        "augmentation_id":
            augmentation_id,

        "experiment":
            experiment,

        "environment":
            environment,

        "n":
            len(values),

        "mean_delta_mse":
            np.mean(values),

        "t_statistic":
            t_stat,

        "p_value":
            p_value,
    })


paired_tests = pd.DataFrame(
    test_records
)


# Benjamini-Hochberg FDR
if not paired_tests.empty:

    order = np.argsort(
        paired_tests[
            "p_value"
        ].to_numpy()
    )

    p = paired_tests[
        "p_value"
    ].to_numpy()

    m = len(p)

    adjusted = np.empty(m)

    ranked = (
        p[order]
        * m
        / np.arange(
            1,
            m + 1
        )
    )

    ranked = np.minimum.accumulate(
        ranked[::-1]
    )[::-1]

    ranked = np.clip(
        ranked,
        0,
        1,
    )

    adjusted[order] = ranked

    paired_tests[
        "p_fdr_bh"
    ] = adjusted


paired_tests.to_csv(
    OUTPUT /
    "paired_environment_tests.csv",
    index=False,
)


# ============================================================
# PLOTS
# ============================================================

print("Creating severity plots...")


for shift_type in [
    "temperature",
    "humidity",
]:

    subset = severity_summary[
        severity_summary[
            "shift_type"
        ] == shift_type
    ]


    for dataset_id in sorted(
        subset[
            "dataset_id"
        ].unique()
    ):

        plot_df = subset[
            subset[
                "dataset_id"
            ] == dataset_id
        ]

        plt.figure(
            figsize=(8, 5)
        )

        for augmentation_id in [
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
        ]:

            temp = plot_df[
                plot_df[
                    "augmentation_id"
                ] == augmentation_id
            ].sort_values(
                "shift_value"
            )

            plt.plot(
                temp["shift_value"],
                temp["mean_mse"],
                marker="o",
                label=augmentation_id,
            )


        plt.xlabel(
            f"{shift_type.title()} shift value"
        )

        plt.ylabel(
            "Mean MSE"
        )

        plt.title(
            f"{dataset_id}: "
            f"{shift_type.title()} "
            f"severity"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT /
            f"{dataset_id}_"
            f"{shift_type}_severity.png",
            dpi=160,
        )

        plt.close()


# ============================================================
# DELTA SEVERITY PLOTS
# ============================================================

print(
    "Creating augmentation-effect plots..."
)


for shift_type in [
    "temperature",
    "humidity",
]:

    subset = (
        severity_delta_summary[
            severity_delta_summary[
                "shift_type"
            ] == shift_type
        ]
    )


    for dataset_id in sorted(
        subset[
            "dataset_id"
        ].unique()
    ):

        plot_df = subset[
            subset[
                "dataset_id"
            ] == dataset_id
        ]

        plt.figure(
            figsize=(8, 5)
        )


        for augmentation_id in [
            "A1",
            "A2",
            "A3",
            "A4",
        ]:

            temp = plot_df[
                plot_df[
                    "augmentation_id"
                ] == augmentation_id
            ].sort_values(
                "shift_value"
            )

            plt.plot(
                temp["shift_value"],
                temp["mean_delta_mse"],
                marker="o",
                label=augmentation_id,
            )


        plt.axhline(
            0,
            linestyle="--",
        )

        plt.xlabel(
            f"{shift_type.title()} shift value"
        )

        plt.ylabel(
            "Mean ΔMSE vs A0"
        )

        plt.title(
            f"{dataset_id}: "
            f"augmentation effect under "
            f"{shift_type} shift"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            OUTPUT /
            f"{dataset_id}_"
            f"{shift_type}_delta_mse.png",
            dpi=160,
        )

        plt.close()


# ============================================================
# ENGINEERING / STATISTICAL REPORT
# ============================================================

overall_fraction_improved = (
    paired[
        "improved_mse"
    ].mean()
)


overall_mean_delta = (
    paired[
        "delta_mse"
    ].mean()
)


report_lines = []

report_lines.append(
    "PHASE 2 CONFIRMATION ANALYSIS REPORT"
)

report_lines.append(
    "=" * 55
)

report_lines.append("")

report_lines.append(
    f"Evaluation records: {len(df)}"
)

report_lines.append(
    f"Unique models: "
    f"{df['model_id'].nunique()}"
)

report_lines.append(
    f"Training compositions: "
    f"{df['dataset_id'].nunique()}"
)

report_lines.append(
    f"Augmentation conditions: "
    f"{df['augmentation_id'].nunique()}"
)

report_lines.append(
    f"Seeds: "
    f"{df['seed'].nunique()}"
)

report_lines.append("")

report_lines.append(
    "Paired augmentation comparison:"
)

report_lines.append(
    f"Overall mean delta MSE: "
    f"{overall_mean_delta:.6f}"
)

report_lines.append(
    f"Fraction of paired evaluations "
    f"with delta MSE < 0: "
    f"{overall_fraction_improved:.4f}"
)

report_lines.append("")

report_lines.append(
    "Interpretation rule:"
)

report_lines.append(
    "delta MSE < 0 means augmentation "
    "improved MSE relative to A0."
)

report_lines.append(
    "delta MSE > 0 means augmentation "
    "increased MSE relative to A0."
)

report_lines.append("")

report_lines.append(
    "Important methodological note:"
)

report_lines.append(
    "Exact lowest-MSE combinations are "
    "descriptive results, not by themselves "
    "evidence of a D x A interaction."
)

report_lines.append(
    "Formal repeated-measures interaction "
    "analysis should follow this descriptive "
    "confirmation analysis."
)

report_lines.append(
    "Temperature and humidity AUC values "
    "must be interpreted separately because "
    "their shift-value scales differ."
)


REPORT_FILE = (
    OUTPUT /
    "phase2_confirmation_report.txt"
)

REPORT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT CHECK
# ============================================================

print()
print("=" * 75)
print("PHASE 2 CONFIRMATION ANALYSIS COMPLETE")
print("=" * 75)

print(
    "Paired D x A summary rows:",
    len(paired_summary)
)

print(
    "Severity summary rows:",
    len(severity_summary)
)

print(
    "AUC summary rows:",
    len(auc_summary)
)

print(
    "Overall fraction improved:",
    round(
        overall_fraction_improved,
        4,
    )
)

print(
    "\nResults saved to:"
)

print(OUTPUT)