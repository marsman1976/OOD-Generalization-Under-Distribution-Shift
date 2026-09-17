from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 3 CONFIRMATION — DESCRIPTIVE / PAIRED ANALYSIS
#
# Central estimand:
#
# delta_mse = MSE(M) - MSE(M0)
#
# delta < 0 : M better than M0
# delta = 0 : same
# delta > 0 : M worse than M0
#
# IMPORTANT:
# This script is DESCRIPTIVE.
# Formal D x M x V inference is performed separately.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "experiment_3"
    / "confirmation"
    / "master_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiment_3"
    / "confirmation"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPECTED DESIGN
# ============================================================


DATASETS = [
    f"D{i:02d}"
    for i in range(10)
]

MODELS = [
    "M0",
    "M1",
    "M2",
]

NONBASELINE_MODELS = [
    "M1",
    "M2",
]

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]


# ============================================================
# LOAD
# ============================================================


def load_results():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} records."
    )

    return df


# ============================================================
# INTEGRITY
# ============================================================


def integrity_check(df):

    print()
    print("=" * 76)
    print("INPUT INTEGRITY CHECK")
    print("=" * 76)

    assert len(df) == 6000

    assert df["run_id"].nunique() == 300

    assert set(
        df["dataset_id"].unique()
    ) == set(DATASETS)

    assert set(
        df["model_config"].unique()
    ) == set(MODELS)

    assert set(
        df["seed"].unique()
    ) == set(SEEDS)

    assert set(
        df["augmentation_id"].unique()
    ) == {"A0"}

    assert df["mse"].notna().all()

    assert df["mae"].notna().all()

    assert np.isfinite(
        df["mse"]
    ).all()

    assert np.isfinite(
        df["mae"]
    ).all()

    duplicates = df.duplicated(
        subset=[
            "run_id",
            "experiment",
            "environment",
        ]
    ).sum()

    assert duplicates == 0

    counts = (
        df.groupby("run_id")
        .size()
    )

    assert (
        counts == 20
    ).all()

    experiment_counts = (
        df["experiment"]
        .value_counts()
    )

    assert (
        experiment_counts["EXP1"]
        == 1200
    )

    assert (
        experiment_counts["EXP1B"]
        == 900
    )

    assert (
        experiment_counts["EXP1C"]
        == 3900
    )

    print("Records:       ", len(df))
    print("Models:        ", df["run_id"].nunique())
    print("Duplicates:    ", duplicates)
    print("Missing MSE:   ", df["mse"].isna().sum())
    print("Integrity:      PASS")


# ============================================================
# BASIC MODEL SUMMARY
#
# This is descriptive only.
# Do NOT interpret as a universal model ranking.
# ============================================================


def model_summary(df):

    summary = (
        df
        .groupby("model_config")
        .agg(
            mean_mse=("mse", "mean"),
            median_mse=("mse", "median"),
            std_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            median_mae=("mae", "median"),
            n=("mse", "size"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "model_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# PAIRED RESULTS
#
# Pair M1/M2 against M0 for exactly the same:
#
# dataset
# seed
# experiment
# environment
# shift type
# severity
# shift value
# ============================================================


def create_paired_results(df):

    key = [
        "dataset_id",
        "seed",
        "experiment",
        "environment",
        "shift_type",
        "severity",
        "shift_value",
    ]

    baseline = (
        df[
            df["model_config"]
            == "M0"
        ]
        [
            key
            + [
                "mse",
                "mae",
                "best_validation_loss",
            ]
        ]
        .rename(
            columns={
                "mse":
                    "mse_M0",
                "mae":
                    "mae_M0",
                "best_validation_loss":
                    "validation_mse_M0",
            }
        )
    )

    paired_frames = []

    for model_config in NONBASELINE_MODELS:

        comparison = (
            df[
                df["model_config"]
                == model_config
            ]
            [
                key
                + [
                    "mse",
                    "mae",
                    "best_validation_loss",
                    "model_parameters",
                ]
            ]
            .rename(
                columns={
                    "mse":
                        "mse_model",
                    "mae":
                        "mae_model",
                    "best_validation_loss":
                        "validation_mse_model",
                }
            )
        )

        comparison[
            "model_config"
        ] = model_config

        paired = comparison.merge(
            baseline,
            on=key,
            how="inner",
            validate="one_to_one",
        )

        paired["delta_mse"] = (
            paired["mse_model"]
            - paired["mse_M0"]
        )

        paired["delta_mae"] = (
            paired["mae_model"]
            - paired["mae_M0"]
        )

        paired[
            "delta_validation_mse"
        ] = (
            paired[
                "validation_mse_model"
            ]
            - paired[
                "validation_mse_M0"
            ]
        )

        paired_frames.append(
            paired
        )

    paired = pd.concat(
        paired_frames,
        ignore_index=True,
    )

    # 6000 total records:
    # M0 = 2000
    # M1 = 2000
    # M2 = 2000
    #
    # paired comparisons:
    # M1 vs M0 = 2000
    # M2 vs M0 = 2000
    #
    # total = 4000

    assert len(paired) == 4000

    assert set(
        paired["model_config"].unique()
    ) == {"M1", "M2"}

    paired.to_csv(
        OUTPUT_DIR
        / "paired_results.csv",
        index=False,
    )

    return paired


# ============================================================
# PAIRED DELTA SUMMARY
# ============================================================


def paired_delta_summary(paired):

    rows = []

    for model_config in NONBASELINE_MODELS:

        x = paired[
            paired["model_config"]
            == model_config
        ]["delta_mse"]

        rows.append({

            "model_config":
                model_config,

            "n":
                len(x),

            "mean_delta_mse":
                x.mean(),

            "median_delta_mse":
                x.median(),

            "std_delta_mse":
                x.std(),

            "min_delta_mse":
                x.min(),

            "max_delta_mse":
                x.max(),

            "fraction_better_than_M0":
                (x < 0).mean(),

            "fraction_worse_than_M0":
                (x > 0).mean(),

            "fraction_equal_M0":
                (x == 0).mean(),
        })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "paired_delta_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# D × M SUMMARY
# ============================================================


def dataset_model_summary(paired):

    summary = (
        paired
        .groupby(
            [
                "dataset_id",
                "model_config",
            ]
        )
        .agg(
            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),
            median_delta_mse=(
                "delta_mse",
                "median",
            ),
            std_delta_mse=(
                "delta_mse",
                "std",
            ),
            fraction_better=(
                "delta_mse",
                lambda x:
                    (x < 0).mean(),
            ),
            n=(
                "delta_mse",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "dataset_model_delta_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# ENVIRONMENT SUMMARY
# ============================================================


def environment_summary(paired):

    summary = (
        paired
        .groupby(
            [
                "experiment",
                "environment",
                "shift_type",
                "model_config",
            ],
            dropna=False,
        )
        .agg(
            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),
            median_delta_mse=(
                "delta_mse",
                "median",
            ),
            std_delta_mse=(
                "delta_mse",
                "std",
            ),
            fraction_better=(
                "delta_mse",
                lambda x:
                    (x < 0).mean(),
            ),
            n=(
                "delta_mse",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "environment_delta_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# SEVERITY SUMMARY
#
# EXP1C only.
# ============================================================


def severity_summary(df):

    severity_df = df[
        df["experiment"]
        == "EXP1C"
    ].copy()

    summary = (
        severity_df
        .groupby(
            [
                "shift_type",
                "severity",
                "shift_value",
                "dataset_id",
                "model_config",
            ]
        )
        .agg(
            mean_mse=(
                "mse",
                "mean",
            ),
            std_mse=(
                "mse",
                "std",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            n=(
                "mse",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "severity_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# SEVERITY DELTA SUMMARY
# ============================================================


def severity_delta_summary(paired):

    severity = paired[
        paired["experiment"]
        == "EXP1C"
    ].copy()

    summary = (
        severity
        .groupby(
            [
                "shift_type",
                "severity",
                "shift_value",
                "dataset_id",
                "model_config",
            ]
        )
        .agg(
            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),
            median_delta_mse=(
                "delta_mse",
                "median",
            ),
            std_delta_mse=(
                "delta_mse",
                "std",
            ),
            fraction_better=(
                "delta_mse",
                lambda x:
                    (x < 0).mean(),
            ),
            n=(
                "delta_mse",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "severity_delta_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# DEGRADATION
#
# For each fitted model:
#
# degradation(V)
# =
# MSE(V) - MSE(V=0)
#
# calculated separately within each shift family.
# ============================================================


def calculate_degradation(df):

    severity = df[
        df["experiment"]
        == "EXP1C"
    ].copy()

    baseline = (
        severity[
            severity["severity"]
            == 0
        ]
        [
            [
                "dataset_id",
                "model_config",
                "seed",
                "shift_type",
                "mse",
            ]
        ]
        .rename(
            columns={
                "mse":
                    "mse_severity0"
            }
        )
    )

    degradation = severity.merge(
        baseline,
        on=[
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
        ],
        how="left",
        validate="many_to_one",
    )

    assert (
        degradation[
            "mse_severity0"
        ]
        .notna()
        .all()
    )

    degradation[
        "degradation_mse"
    ] = (
        degradation["mse"]
        - degradation["mse_severity0"]
    )

    degradation.to_csv(
        OUTPUT_DIR
        / "degradation.csv",
        index=False,
    )

    return degradation


# ============================================================
# ROBUSTNESS AUC
#
# Integrate degradation across the actual shift values.
#
# Lower degradation AUC = less accumulated degradation.
#
# np.trapezoid is preferred on newer NumPy.
# np.trapz fallback supports older NumPy.
# ============================================================


def trapezoid_auc(y, x):

    if hasattr(
        np,
        "trapezoid"
    ):

        return np.trapezoid(
            y,
            x
        )

    return np.trapz(
        y,
        x
    )


def calculate_auc(degradation):

    rows = []

    group_cols = [
        "dataset_id",
        "model_config",
        "seed",
        "shift_type",
    ]

    for keys, group in (
        degradation
        .groupby(group_cols)
    ):

        (
            dataset_id,
            model_config,
            seed,
            shift_type,
        ) = keys

        group = (
            group
            .sort_values(
                "shift_value"
            )
        )

        x = (
            group[
                "shift_value"
            ]
            .to_numpy(
                dtype=float
            )
        )

        y = (
            group[
                "degradation_mse"
            ]
            .to_numpy(
                dtype=float
            )
        )

        auc = trapezoid_auc(
            y,
            x
        )

        rows.append({

            "dataset_id":
                dataset_id,

            "model_config":
                model_config,

            "seed":
                seed,

            "shift_type":
                shift_type,

            "degradation_auc":
                auc,
        })

    auc_df = pd.DataFrame(
        rows
    )

    # 10 D × 3 M × 10 seeds × 2 shift families
    assert len(auc_df) == 600

    auc_df.to_csv(
        OUTPUT_DIR
        / "robustness_auc_per_model.csv",
        index=False,
    )

    return auc_df


# ============================================================
# AUC SUMMARY
# ============================================================


def auc_summary(auc_df):

    summary = (
        auc_df
        .groupby(
            [
                "shift_type",
                "dataset_id",
                "model_config",
            ]
        )
        .agg(
            mean_degradation_auc=(
                "degradation_auc",
                "mean",
            ),
            median_degradation_auc=(
                "degradation_auc",
                "median",
            ),
            std_degradation_auc=(
                "degradation_auc",
                "std",
            ),
            n=(
                "degradation_auc",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "robustness_auc_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# PAIRED AUC DELTA
#
# delta_auc = AUC(M) - AUC(M0)
#
# negative:
# less accumulated degradation than M0
#
# positive:
# more accumulated degradation than M0
# ============================================================


def auc_delta(auc_df):

    key = [
        "dataset_id",
        "seed",
        "shift_type",
    ]

    baseline = (
        auc_df[
            auc_df["model_config"]
            == "M0"
        ]
        [
            key
            + [
                "degradation_auc"
            ]
        ]
        .rename(
            columns={
                "degradation_auc":
                    "auc_M0"
            }
        )
    )

    comparisons = []

    for model_config in NONBASELINE_MODELS:

        current = (
            auc_df[
                auc_df["model_config"]
                == model_config
            ]
            [
                key
                + [
                    "degradation_auc"
                ]
            ]
            .rename(
                columns={
                    "degradation_auc":
                        "auc_model"
                }
            )
        )

        current[
            "model_config"
        ] = model_config

        paired_auc = current.merge(
            baseline,
            on=key,
            how="inner",
            validate="one_to_one",
        )

        paired_auc[
            "delta_auc"
        ] = (
            paired_auc[
                "auc_model"
            ]
            - paired_auc[
                "auc_M0"
            ]
        )

        comparisons.append(
            paired_auc
        )

    result = pd.concat(
        comparisons,
        ignore_index=True,
    )

    # 10 D × 10 seeds × 2 shifts × 2 comparisons
    assert len(result) == 400

    result.to_csv(
        OUTPUT_DIR
        / "robustness_auc_delta.csv",
        index=False,
    )

    return result


# ============================================================
# ID / OOD TRADE-OFF
#
# EXP1 normal is treated as ID reference.
# Other EXP1/EXP1B environments are summarized as shifted.
#
# This is descriptive.
# ============================================================


def id_ood_tradeoff(paired):

    exp = paired[
        paired["experiment"]
        .isin(
            [
                "EXP1",
                "EXP1B",
            ]
        )
    ].copy()

    exp["environment_class"] = np.where(
        (
            (exp["experiment"] == "EXP1")
            &
            (exp["environment"] == "normal")
        ),
        "ID",
        "SHIFTED",
    )

    summary = (
        exp
        .groupby(
            [
                "dataset_id",
                "model_config",
                "environment_class",
            ]
        )
        .agg(
            mean_delta_mse=(
                "delta_mse",
                "mean",
            ),
            median_delta_mse=(
                "delta_mse",
                "median",
            ),
            fraction_better=(
                "delta_mse",
                lambda x:
                    (x < 0).mean(),
            ),
            n=(
                "delta_mse",
                "size",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR
        / "id_ood_tradeoff.csv",
        index=False,
    )

    return summary


# ============================================================
# PLOTS — DELTA VS SEVERITY
# ============================================================


def plot_delta_by_severity(
    severity_delta
):

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        shift_df = severity_delta[
            severity_delta[
                "shift_type"
            ]
            == shift_type
        ]

        for model_config in NONBASELINE_MODELS:

            model_df = shift_df[
                shift_df[
                    "model_config"
                ]
                == model_config
            ]

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            for dataset_id in DATASETS:

                d = (
                    model_df[
                        model_df[
                            "dataset_id"
                        ]
                        == dataset_id
                    ]
                    .sort_values(
                        "shift_value"
                    )
                )

                ax.plot(
                    d["shift_value"],
                    d["mean_delta_mse"],
                    marker="o",
                    label=dataset_id,
                )

            ax.axhline(
                0,
                linewidth=1,
            )

            ax.set_xlabel(
                "Shift value"
            )

            ax.set_ylabel(
                "Mean ΔMSE vs M0"
            )

            ax.set_title(
                f"{model_config} vs M0 — "
                f"{shift_type} shift"
            )

            ax.legend(
                ncol=2,
                fontsize=8,
            )

            fig.tight_layout()

            output = (
                PLOT_DIR
                / (
                    f"delta_"
                    f"{shift_type}_"
                    f"{model_config}.png"
                )
            )

            fig.savefig(
                output,
                dpi=180,
            )

            plt.close(fig)


# ============================================================
# PLOTS — RAW SEVERITY TRAJECTORIES
# ============================================================


def plot_raw_severity(
    severity_summary_df
):

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        shift_df = (
            severity_summary_df[
                severity_summary_df[
                    "shift_type"
                ]
                == shift_type
            ]
        )

        for dataset_id in DATASETS:

            d = shift_df[
                shift_df[
                    "dataset_id"
                ]
                == dataset_id
            ]

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            for model_config in MODELS:

                m = (
                    d[
                        d[
                            "model_config"
                        ]
                        == model_config
                    ]
                    .sort_values(
                        "shift_value"
                    )
                )

                ax.plot(
                    m["shift_value"],
                    m["mean_mse"],
                    marker="o",
                    label=model_config,
                )

            ax.set_xlabel(
                "Shift value"
            )

            ax.set_ylabel(
                "Mean MSE"
            )

            ax.set_title(
                f"{dataset_id} — "
                f"{shift_type} shift"
            )

            ax.legend()

            fig.tight_layout()

            output = (
                PLOT_DIR
                / (
                    f"raw_"
                    f"{shift_type}_"
                    f"{dataset_id}.png"
                )
            )

            fig.savefig(
                output,
                dpi=180,
            )

            plt.close(fig)


# ============================================================
# TEXT REPORT
# ============================================================


def write_report(
    df,
    paired,
    delta_summary,
    auc_delta_df,
):

    report_file = (
        OUTPUT_DIR
        / "phase3_confirmation_report.txt"
    )

    lines = []

    lines.append(
        "PHASE 3 CONFIRMATION — DESCRIPTIVE REPORT"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        f"Evaluation records: {len(df)}"
    )

    lines.append(
        f"Trained models: "
        f"{df['run_id'].nunique()}"
    )

    lines.append(
        f"Paired comparison records: "
        f"{len(paired)}"
    )

    lines.append("")

    lines.append(
        "Central estimand:"
    )

    lines.append(
        "delta_mse = MSE(model) - MSE(M0)"
    )

    lines.append(
        "Negative delta indicates lower MSE than M0."
    )

    lines.append("")

    lines.append(
        "OVERALL PAIRED DELTA SUMMARY"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        delta_summary.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "ROBUSTNESS AUC DELTA SUMMARY"
    )

    lines.append(
        "-" * 72
    )

    auc_report = (
        auc_delta_df
        .groupby(
            [
                "shift_type",
                "model_config",
            ]
        )
        .agg(
            mean_delta_auc=(
                "delta_auc",
                "mean",
            ),
            median_delta_auc=(
                "delta_auc",
                "median",
            ),
            fraction_lower_auc=(
                "delta_auc",
                lambda x:
                    (x < 0).mean(),
            ),
            n=(
                "delta_auc",
                "size",
            ),
        )
        .reset_index()
    )

    lines.append(
        auc_report.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "IMPORTANT INTERPRETATION NOTE"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "These summaries are descriptive."
    )

    lines.append(
        "Do not infer D x M x V interaction significance "
        "from these tables alone."
    )

    lines.append(
        "Formal clustered interaction analysis is required."
    )

    lines.append(
        "Pilot results were not used to redesign "
        "M1 or M2."
    )

    lines.append("")

    lines.append(
        "NEXT:"
    )

    lines.append(
        "Run formal Phase 3 D x M x V interaction analysis "
        "separately for temperature and humidity."
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    return report_file


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 76)
    print("PHASE 3 CONFIRMATION ANALYSIS")
    print("=" * 76)

    df = load_results()

    integrity_check(df)

    print()
    print("Creating model summary...")

    model_summary_df = model_summary(
        df
    )

    print(
        model_summary_df.to_string(
            index=False
        )
    )


    print()
    print("Creating paired M1/M2 vs M0 results...")

    paired = create_paired_results(
        df
    )

    print(
        "Paired records:",
        len(paired)
    )


    print()
    print("Creating paired delta summary...")

    delta_summary = paired_delta_summary(
        paired
    )

    print(
        delta_summary.to_string(
            index=False
        )
    )


    print()
    print("Creating D x M descriptive summary...")

    dataset_model_summary(
        paired
    )


    print(
        "Creating environment summary..."
    )

    environment_summary(
        paired
    )


    print(
        "Creating severity summaries..."
    )

    severity_summary_df = severity_summary(
        df
    )

    severity_delta = severity_delta_summary(
        paired
    )


    print(
        "Calculating degradation trajectories..."
    )

    degradation = calculate_degradation(
        df
    )


    print(
        "Calculating robustness AUC..."
    )

    auc_df = calculate_auc(
        degradation
    )

    auc_summary(
        auc_df
    )


    print(
        "Calculating paired AUC deltas..."
    )

    auc_delta_df = auc_delta(
        auc_df
    )


    print(
        "Calculating ID/OOD trade-off summary..."
    )

    id_ood_tradeoff(
        paired
    )


    print(
        "Creating plots..."
    )

    plot_delta_by_severity(
        severity_delta
    )

    plot_raw_severity(
        severity_summary_df
    )


    report_file = write_report(
        df,
        paired,
        delta_summary,
        auc_delta_df,
    )


    print()
    print("=" * 76)
    print("PHASE 3 DESCRIPTIVE ANALYSIS COMPLETE")
    print("=" * 76)

    print(
        f"Analysis directory:\n{OUTPUT_DIR}"
    )

    print(
        f"\nReport:\n{report_file}"
    )

    print()
    print(
        "IMPORTANT: These are descriptive results."
    )

    print(
        "Do not make the final D x M x V "
        "scientific conclusion yet."
    )


if __name__ == "__main__":
    main()