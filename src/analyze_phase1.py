from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    t,
    spearmanr,
)


# ============================================================
# PATHS
# ============================================================

EXP1_FILE = Path("results/runs.csv")
EXP1B_FILE = Path("results/experiment_1b/runs_1b.csv")
EXP1C_FILE = Path("results/experiment_1c/runs_1c.csv")

OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    exp1 = pd.read_csv(EXP1_FILE)
    exp1b = pd.read_csv(EXP1B_FILE)
    exp1c = pd.read_csv(EXP1C_FILE)

    print("Experiment 1 :", exp1.shape)
    print("Experiment 1B:", exp1b.shape)
    print("Experiment 1C:", exp1c.shape)

    assert len(exp1) == 120
    assert len(exp1b) == 90
    assert len(exp1c) == 390

    return exp1, exp1b, exp1c


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def summarize(df, group_columns):

    summary = (
        df.groupby(group_columns)
        .agg(
            n=("mse", "count"),
            mean_mse=("mse", "mean"),
            sd_mse=("mse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
        )
        .reset_index()
    )

    summary["se_mse"] = (
        summary["sd_mse"]
        / np.sqrt(summary["n"])
    )

    # t critical value for each group
    summary["t_critical"] = summary["n"].apply(
        lambda n:
        t.ppf(0.975, df=n - 1)
        if n > 1 else np.nan
    )

    summary["ci95_low"] = (
        summary["mean_mse"]
        - summary["t_critical"]
        * summary["se_mse"]
    )

    summary["ci95_high"] = (
        summary["mean_mse"]
        + summary["t_critical"]
        * summary["se_mse"]
    )

    return summary


# ============================================================
# EXPERIMENT 1
# ============================================================

def analyze_experiment1(exp1):

    summary = summarize(
        exp1,
        [
            "dataset_id",
            "shift",
        ],
    )

    summary.to_csv(
        OUTPUT_DIR / "experiment1_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# EXPERIMENT 1B
# ============================================================

def analyze_experiment1b(exp1b):

    summary = summarize(
        exp1b,
        [
            "dataset_id",
            "shift",
        ],
    )

    summary.to_csv(
        OUTPUT_DIR / "experiment1b_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# EXPERIMENT 1C SUMMARY
# ============================================================

def analyze_severity(exp1c):

    summary = summarize(
        exp1c,
        [
            "dataset_id",
            "shift_type",
            "severity",
            "shift_value",
        ],
    )

    summary.to_csv(
        OUTPUT_DIR / "severity_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# WINNER AT EACH SEVERITY
# ============================================================

def calculate_winners(severity_summary):

    index = (
        severity_summary
        .groupby(
            [
                "shift_type",
                "severity",
            ]
        )["mean_mse"]
        .idxmin()
    )

    winners = (
        severity_summary
        .loc[index]
        [
            [
                "shift_type",
                "severity",
                "shift_value",
                "dataset_id",
                "mean_mse",
                "sd_mse",
            ]
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
        OUTPUT_DIR / "severity_winners.csv",
        index=False,
    )

    return winners


# ============================================================
# DEGRADATION
# ============================================================

def calculate_degradation(severity_summary):

    result = []

    for shift_type in (
        severity_summary["shift_type"].unique()
    ):

        subset = severity_summary[
            severity_summary["shift_type"]
            == shift_type
        ]

        for dataset_id in (
            subset["dataset_id"].unique()
        ):

            data = subset[
                subset["dataset_id"]
                == dataset_id
            ].sort_values("severity")

            baseline = data[
                data["severity"] == 0
            ]["mean_mse"].iloc[0]

            for _, row in data.iterrows():

                absolute = (
                    row["mean_mse"]
                    - baseline
                )

                relative = (
                    absolute / baseline
                    if baseline != 0
                    else np.nan
                )

                result.append({

                    "dataset_id":
                        dataset_id,

                    "shift_type":
                        shift_type,

                    "severity":
                        row["severity"],

                    "shift_value":
                        row["shift_value"],

                    "mean_mse":
                        row["mean_mse"],

                    "baseline_mse":
                        baseline,

                    "absolute_degradation":
                        absolute,

                    "relative_degradation":
                        relative,
                })

    degradation = pd.DataFrame(result)

    degradation.to_csv(
        OUTPUT_DIR / "degradation.csv",
        index=False,
    )

    return degradation


# ============================================================
# RANK CORRELATIONS
# ============================================================

def calculate_rank_correlations(
    severity_summary
):

    results = []

    for shift_type in (
        severity_summary["shift_type"].unique()
    ):

        subset = severity_summary[
            severity_summary["shift_type"]
            == shift_type
        ]

        baseline = (
            subset[
                subset["severity"] == 0
            ]
            .set_index("dataset_id")
            ["mean_mse"]
        )

        severities = sorted(
            subset["severity"].unique()
        )

        for severity in severities:

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

            rho, p_value = spearmanr(
                baseline.loc[common],
                current.loc[common],
            )

            results.append({

                "shift_type":
                    shift_type,

                "severity":
                    severity,

                "spearman_rho":
                    rho,

                "p_value":
                    p_value,
            })

    correlations = pd.DataFrame(results)

    correlations.to_csv(
        OUTPUT_DIR /
        "rank_correlations.csv",
        index=False,
    )

    return correlations


# ============================================================
# ROBUSTNESS AUC
# ============================================================

def calculate_auc(
    severity_summary
):

    results = []

    for shift_type in (
        severity_summary["shift_type"].unique()
    ):

        subset = severity_summary[
            severity_summary["shift_type"]
            == shift_type
        ]

        for dataset_id in (
            subset["dataset_id"].unique()
        ):

            data = (
                subset[
                    subset["dataset_id"]
                    == dataset_id
                ]
                .sort_values("shift_value")
            )

            x = (
                data["shift_value"]
                .to_numpy()
            )

            y = (
                data["mean_mse"]
                .to_numpy()
            )

            auc = np.trapezoid(
                y,
                x,
           )

            normalized_auc = (
                auc /
                (x.max() - x.min())
            )

            results.append({

                "dataset_id":
                    dataset_id,

                "shift_type":
                    shift_type,

                "auc":
                    auc,

                "normalized_auc":
                    normalized_auc,
            })

    auc_df = pd.DataFrame(results)

    auc_df["robustness_rank"] = (
        auc_df
        .groupby("shift_type")
        ["auc"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    auc_df = auc_df.sort_values(
        [
            "shift_type",
            "robustness_rank",
        ]
    )

    auc_df.to_csv(
        OUTPUT_DIR /
        "robustness_auc.csv",
        index=False,
    )

    return auc_df


# ============================================================
# TEMPERATURE CURVES
# ============================================================

def plot_temperature(
    severity_summary
):

    data = severity_summary[
        severity_summary["shift_type"]
        == "TEMPERATURE"
    ]

    plt.figure(
        figsize=(10, 7)
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

    plt.xlabel(
        "Air temperature (°C)"
    )

    plt.ylabel(
        "Mean test MSE"
    )

    plt.title(
        "Temperature Shift-Severity Curves"
    )

    plt.legend(
        ncol=2
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "temperature_curves.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# HUMIDITY CURVES
# ============================================================

def plot_humidity(
    severity_summary
):

    data = severity_summary[
        severity_summary["shift_type"]
        == "HUMIDITY"
    ]

    plt.figure(
        figsize=(10, 7)
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

    plt.xlabel(
        "Relative humidity (%)"
    )

    plt.ylabel(
        "Mean test MSE"
    )

    plt.title(
        "Humidity Shift-Severity Curves"
    )

    plt.legend(
        ncol=2
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "humidity_curves.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# TEXT REPORT
# ============================================================

def create_report(
    winners,
    correlations,
    auc_df,
):

    report_file = (
        OUTPUT_DIR /
        "phase1_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 1 PILOT ANALYSIS\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Results are based on 3 training seeds "
            "and should be interpreted as pilot evidence.\n\n"
        )

        f.write(
            "SEVERITY WINNERS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            winners.to_string(index=False)
        )

        f.write("\n\n")

        f.write(
            "RANK CORRELATIONS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            correlations.to_string(index=False)
        )

        f.write("\n\n")

        f.write(
            "ROBUSTNESS AUC\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            auc_df.to_string(index=False)
        )

        f.write("\n")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PHASE 1 ANALYSIS")
    print("=" * 70)

    exp1, exp1b, exp1c = (
        load_data()
    )

    exp1_summary = (
        analyze_experiment1(exp1)
    )

    exp1b_summary = (
        analyze_experiment1b(exp1b)
    )

    severity_summary = (
        analyze_severity(exp1c)
    )

    winners = calculate_winners(
        severity_summary
    )

    degradation = (
        calculate_degradation(
            severity_summary
        )
    )

    correlations = (
        calculate_rank_correlations(
            severity_summary
        )
    )

    auc_df = calculate_auc(
        severity_summary
    )

    plot_temperature(
        severity_summary
    )

    plot_humidity(
        severity_summary
    )

    create_report(
        winners,
        correlations,
        auc_df,
    )

    print()
    print("Analysis complete.")
    print()

    print(
        "Experiment 1 summary:",
        len(exp1_summary),
        "rows"
    )

    print(
        "Experiment 1B summary:",
        len(exp1b_summary),
        "rows"
    )

    print(
        "Severity summary:",
        len(severity_summary),
        "rows"
    )

    print(
        "Degradation records:",
        len(degradation)
    )

    print()

    print(
        "Files saved in:",
        OUTPUT_DIR
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()