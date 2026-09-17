from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, t


# ============================================================
# PATHS
# ============================================================

MASTER_FILE = Path(
    "results/phase1_confirmation/master_results.csv"
)

PILOT_FILE = Path(
    "results/analysis/severity_summary.csv"
)

OUTPUT_DIR = Path(
    "results/phase1_confirmation/analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

def load_data():

    df = pd.read_csv(MASTER_FILE)

    print("Master shape:", df.shape)

    if len(df) != 2000:
        raise ValueError(
            f"Expected 2000 rows, got {len(df)}"
        )

    required = {
        "dataset_id",
        "seed",
        "experiment",
        "shift_type",
        "severity",
        "shift_value",
        "mse",
        "mae",
        "validation_mse",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def add_statistics(grouped):

    grouped["se_mse"] = (
        grouped["sd_mse"]
        / np.sqrt(grouped["n"])
    )

    grouped["t_critical"] = grouped["n"].apply(
        lambda n:
        t.ppf(0.975, n - 1)
        if n > 1 else np.nan
    )

    grouped["ci95_low"] = (
        grouped["mean_mse"]
        - grouped["t_critical"]
        * grouped["se_mse"]
    )

    grouped["ci95_high"] = (
        grouped["mean_mse"]
        + grouped["t_critical"]
        * grouped["se_mse"]
    )

    return grouped


# ============================================================
# EXPERIMENT 1 + 1B SUMMARY
# ============================================================

def general_summary(df):

    data = df[
        df["experiment"].isin(
            ["EXP1", "EXP1B"]
        )
    ]

    summary = (
        data.groupby(
            [
                "experiment",
                "dataset_id",
                "shift_type",
            ]
        )
        .agg(
            n=("mse", "count"),
            mean_mse=("mse", "mean"),
            sd_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
        )
        .reset_index()
    )

    summary = add_statistics(summary)

    summary.to_csv(
        OUTPUT_DIR /
        "confirmation_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# SEVERITY SUMMARY
# ============================================================

def severity_summary(df):

    data = df[
        df["experiment"] == "EXP1C"
    ]

    summary = (
        data.groupby(
            [
                "dataset_id",
                "shift_type",
                "severity",
                "shift_value",
            ]
        )
        .agg(
            n=("mse", "count"),
            mean_mse=("mse", "mean"),
            sd_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
        )
        .reset_index()
    )

    summary = add_statistics(summary)

    summary.to_csv(
        OUTPUT_DIR /
        "severity_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# WINNERS
# ============================================================

def severity_winners(summary):

    idx = (
        summary
        .groupby(
            [
                "shift_type",
                "severity",
            ]
        )["mean_mse"]
        .idxmin()
    )

    winners = (
        summary
        .loc[
            idx,
            [
                "shift_type",
                "severity",
                "shift_value",
                "dataset_id",
                "mean_mse",
                "sd_mse",
                "ci95_low",
                "ci95_high",
            ],
        ]
        .sort_values(
            [
                "shift_type",
                "severity",
            ]
        )
        .reset_index(drop=True)
    )

    winners.to_csv(
        OUTPUT_DIR /
        "severity_winners.csv",
        index=False,
    )

    return winners


# ============================================================
# DEGRADATION
#
# IMPORTANT:
# calculate degradation WITHIN each trained model first.
# This preserves the repeated-measures relationship.
# ============================================================

def degradation_analysis(df):

    data = df[
        df["experiment"] == "EXP1C"
    ].copy()

    records = []

    for (
        dataset_id,
        seed,
        shift_type
    ), group in data.groupby(
        [
            "dataset_id",
            "seed",
            "shift_type",
        ]
    ):

        group = group.sort_values(
            "severity"
        )

        baseline_rows = group[
            group["severity"] == 0
        ]

        if len(baseline_rows) != 1:
            raise ValueError(
                "Expected exactly one baseline "
                f"for {dataset_id}, "
                f"seed={seed}, "
                f"{shift_type}"
            )

        baseline = (
            baseline_rows["mse"].iloc[0]
        )

        for _, row in group.iterrows():

            absolute = (
                row["mse"] - baseline
            )

            relative = (
                absolute / baseline
                if baseline != 0
                else np.nan
            )

            records.append({
                "dataset_id":
                    dataset_id,

                "seed":
                    seed,

                "shift_type":
                    shift_type,

                "severity":
                    int(row["severity"]),

                "shift_value":
                    row["shift_value"],

                "mse":
                    row["mse"],

                "baseline_mse":
                    baseline,

                "absolute_degradation":
                    absolute,

                "relative_degradation":
                    relative,
            })

    degradation = pd.DataFrame(records)

    degradation.to_csv(
        OUTPUT_DIR /
        "degradation.csv",
        index=False,
    )

    return degradation


# ============================================================
# AUC
#
# IMPORTANT:
# calculate AUC PER TRAINED MODEL first,
# then summarize across seeds.
# ============================================================

def robustness_auc(df):

    data = df[
        df["experiment"] == "EXP1C"
    ].copy()

    records = []

    for (
        dataset_id,
        seed,
        shift_type
    ), group in data.groupby(
        [
            "dataset_id",
            "seed",
            "shift_type",
        ]
    ):

        group = group.sort_values(
            "shift_value"
        )

        x = group[
            "shift_value"
        ].to_numpy()

        y = group[
            "mse"
        ].to_numpy()

        # Manual trapezoidal integration.
        # Compatible across NumPy versions.
        auc = np.sum(
            (x[1:] - x[:-1])
            * (y[1:] + y[:-1])
            / 2.0
        )

        width = (
            x.max() - x.min()
        )

        normalized_auc = (
            auc / width
        )

        records.append({
            "dataset_id":
                dataset_id,

            "seed":
                seed,

            "shift_type":
                shift_type,

            "auc":
                auc,

            "normalized_auc":
                normalized_auc,
        })

    individual_auc = pd.DataFrame(
        records
    )

    individual_auc.to_csv(
        OUTPUT_DIR /
        "robustness_auc_per_seed.csv",
        index=False,
    )

    summary = (
        individual_auc
        .groupby(
            [
                "dataset_id",
                "shift_type",
            ]
        )
        .agg(
            n=("auc", "count"),
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
            mean_normalized_auc=(
                "normalized_auc",
                "mean",
            ),
            sd_normalized_auc=(
                "normalized_auc",
                "std",
            ),
        )
        .reset_index()
    )

    summary["se_auc"] = (
        summary["sd_auc"]
        / np.sqrt(summary["n"])
    )

    summary["t_critical"] = (
        summary["n"].apply(
            lambda n:
            t.ppf(0.975, n - 1)
            if n > 1 else np.nan
        )
    )

    summary["ci95_low"] = (
        summary["mean_auc"]
        - summary["t_critical"]
        * summary["se_auc"]
    )

    summary["ci95_high"] = (
        summary["mean_auc"]
        + summary["t_critical"]
        * summary["se_auc"]
    )

    summary["robustness_rank"] = (
        summary
        .groupby("shift_type")
        ["mean_auc"]
        .rank(
            ascending=True,
            method="min",
        )
    )

    summary = summary.sort_values(
        [
            "shift_type",
            "robustness_rank",
        ]
    )

    summary.to_csv(
        OUTPUT_DIR /
        "robustness_auc.csv",
        index=False,
    )

    return individual_auc, summary


# ============================================================
# RANK CORRELATIONS
# ============================================================

def rank_correlations(summary):

    records = []

    for shift_type in sorted(
        summary["shift_type"].unique()
    ):

        subset = summary[
            summary["shift_type"]
            == shift_type
        ]

        baseline = (
            subset[
                subset["severity"] == 0
            ]
            .set_index("dataset_id")
            ["mean_mse"]
        )

        for severity in sorted(
            subset["severity"].unique()
        ):

            current = (
                subset[
                    subset["severity"]
                    == severity
                ]
                .set_index("dataset_id")
                ["mean_mse"]
            )

            common = (
                baseline.index
                .intersection(current.index)
            )

            rho, p = spearmanr(
                baseline.loc[common],
                current.loc[common],
            )

            records.append({
                "shift_type":
                    shift_type,

                "severity":
                    int(severity),

                "n_compositions":
                    len(common),

                "spearman_rho":
                    rho,

                "p_value":
                    p,
            })

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_DIR /
        "rank_correlations.csv",
        index=False,
    )

    return result


# ============================================================
# PILOT VS CONFIRMATION
# ============================================================

def compare_with_pilot(
    confirmation_summary
):

    if not PILOT_FILE.exists():

        print(
            "Pilot severity summary not found."
        )

        return None

    pilot = pd.read_csv(
        PILOT_FILE
    )

    keys = [
        "dataset_id",
        "shift_type",
        "severity",
        "shift_value",
    ]

    pilot = pilot[
        keys + ["mean_mse"]
    ].rename(
        columns={
            "mean_mse":
                "pilot_mean_mse"
        }
    )

    confirmation = (
        confirmation_summary[
            keys + ["mean_mse"]
        ]
        .rename(
            columns={
                "mean_mse":
                    "confirmation_mean_mse"
            }
        )
    )

    comparison = pilot.merge(
        confirmation,
        on=keys,
        how="inner",
    )

    comparison[
        "absolute_difference"
    ] = (
        comparison[
            "confirmation_mean_mse"
        ]
        - comparison[
            "pilot_mean_mse"
        ]
    )

    comparison[
        "percent_difference"
    ] = (
        100
        * comparison[
            "absolute_difference"
        ]
        / comparison[
            "pilot_mean_mse"
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR /
        "pilot_vs_confirmation.csv",
        index=False,
    )

    return comparison


# ============================================================
# PLOT SEVERITY CURVES
# ============================================================

def plot_curves(
    summary,
    shift_type,
    filename,
):

    data = summary[
        summary["shift_type"]
        == shift_type
    ]

    plt.figure(
        figsize=(11, 7)
    )

    for dataset_id in sorted(
        data["dataset_id"].unique()
    ):

        subset = (
            data[
                data["dataset_id"]
                == dataset_id
            ]
            .sort_values("shift_value")
        )

        plt.plot(
            subset["shift_value"],
            subset["mean_mse"],
            marker="o",
            label=dataset_id,
        )

    if shift_type == "TEMPERATURE":

        plt.xlabel(
            "Air temperature (°C)"
        )

    else:

        plt.xlabel(
            "Relative humidity (%)"
        )

    plt.ylabel(
        "Mean test MSE"
    )

    plt.title(
        f"{shift_type.title()} "
        "Shift-Severity Curves"
    )

    plt.legend(
        ncol=2
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# DEGRADATION PLOTS
# ============================================================

def plot_degradation(
    degradation,
    shift_type,
    filename,
):

    data = degradation[
        degradation["shift_type"]
        == shift_type
    ]

    summary = (
        data.groupby(
            [
                "dataset_id",
                "shift_value",
            ]
        )
        ["absolute_degradation"]
        .mean()
        .reset_index()
    )

    plt.figure(
        figsize=(11, 7)
    )

    for dataset_id in sorted(
        summary["dataset_id"].unique()
    ):

        subset = (
            summary[
                summary["dataset_id"]
                == dataset_id
            ]
            .sort_values("shift_value")
        )

        plt.plot(
            subset["shift_value"],
            subset[
                "absolute_degradation"
            ],
            marker="o",
            label=dataset_id,
        )

    if shift_type == "TEMPERATURE":

        plt.xlabel(
            "Air temperature (°C)"
        )

    else:

        plt.xlabel(
            "Relative humidity (%)"
        )

    plt.ylabel(
        "MSE increase from severity 0"
    )

    plt.title(
        f"{shift_type.title()} "
        "Absolute Degradation"
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# RANK TRAJECTORY
# ============================================================

def plot_rank_trajectory(
    summary,
    shift_type,
    filename,
):

    data = summary[
        summary["shift_type"]
        == shift_type
    ].copy()

    data["rank"] = (
        data.groupby("severity")
        ["mean_mse"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    for dataset_id in sorted(
        data["dataset_id"].unique()
    ):

        subset = (
            data[
                data["dataset_id"]
                == dataset_id
            ]
            .sort_values("severity")
        )

        plt.plot(
            subset["severity"],
            subset["rank"],
            marker="o",
            label=dataset_id,
        )

    plt.xlabel(
        "Severity level"
    )

    plt.ylabel(
        "Performance rank (1 = lowest MSE)"
    )

    plt.title(
        f"{shift_type.title()} "
        "Composition Rank Trajectory"
    )

    # Rank 1 at top
    plt.gca().invert_yaxis()

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
    )

    plt.close()


# ============================================================
# REPORT
# ============================================================

def create_report(
    df,
    winners,
    correlations,
    auc_summary,
    comparison,
):

    report = (
        OUTPUT_DIR /
        "phase1_confirmation_report.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 1 EXPANDED-SEED "
            "CONFIRMATION ANALYSIS\n"
        )

        f.write(
            "=" * 75 + "\n\n"
        )

        f.write(
            "DESIGN\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            f"Master observations: "
            f"{len(df)}\n"
        )

        f.write(
            "Training compositions: "
            f"{df['dataset_id'].nunique()}\n"
        )

        f.write(
            "Training seeds: "
            f"{df['seed'].nunique()}\n"
        )

        f.write(
            "This is an expanded-seed "
            "confirmation, not a fully "
            "independent replication, "
            "because seeds 11, 22 and 33 "
            "were also used in the pilot.\n\n"
        )

        f.write(
            "SEVERITY WINNERS\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            winners.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "RANK CORRELATIONS\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            correlations.to_string(
                index=False
            )
        )

        f.write("\n\n")

        f.write(
            "ROBUSTNESS AUC\n"
        )

        f.write(
            "-" * 75 + "\n"
        )

        f.write(
            auc_summary.to_string(
                index=False
            )
        )

        f.write("\n\n")

        if comparison is not None:

            f.write(
                "PILOT VS CONFIRMATION\n"
            )

            f.write(
                "-" * 75 + "\n"
            )

            f.write(
                "Comparison table saved as "
                "pilot_vs_confirmation.csv\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "PHASE 1 CONFIRMATION ANALYSIS"
    )
    print("=" * 75)

    df = load_data()

    confirmation = (
        general_summary(df)
    )

    severity = (
        severity_summary(df)
    )

    winners = (
        severity_winners(severity)
    )

    degradation = (
        degradation_analysis(df)
    )

    _, auc_summary = (
        robustness_auc(df)
    )

    correlations = (
        rank_correlations(severity)
    )

    comparison = (
        compare_with_pilot(
            severity
        )
    )

    # ---------------- PLOTS ----------------

    plot_curves(
        severity,
        "TEMPERATURE",
        "temperature_curves.png",
    )

    plot_curves(
        severity,
        "HUMIDITY",
        "humidity_curves.png",
    )

    plot_degradation(
        degradation,
        "TEMPERATURE",
        "temperature_degradation.png",
    )

    plot_degradation(
        degradation,
        "HUMIDITY",
        "humidity_degradation.png",
    )

    plot_rank_trajectory(
        severity,
        "TEMPERATURE",
        "rank_trajectory_temperature.png",
    )

    plot_rank_trajectory(
        severity,
        "HUMIDITY",
        "rank_trajectory_humidity.png",
    )

    create_report(
        df,
        winners,
        correlations,
        auc_summary,
        comparison,
    )

    print()
    print("Analysis completed successfully.")
    print()

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print()
    print("=" * 75)


if __name__ == "__main__":
    main()