from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "experiment_2" /
    "pilot" / "runs_phase2_pilot.csv"
)

OUTPUT = (
    ROOT / "results" / "experiment_2" /
    "pilot" / "analysis"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


def save_table(df, name):
    path = OUTPUT / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def integrity_check(df):

    print("\n" + "=" * 70)
    print("1. INTEGRITY CHECK")
    print("=" * 70)

    print("Rows:", len(df))
    print("Models:", df["model_id"].nunique())

    duplicates = df.duplicated(
        subset=["model_id", "experiment", "environment"]
    ).sum()

    print("Duplicates:", duplicates)

    assert len(df) == 900
    assert df["model_id"].nunique() == 45
    assert duplicates == 0

    assert not df["mse"].isna().any()
    assert not df["mae"].isna().any()

    print("Integrity: PASS")


def compute_paired_delta(df):
    """
    Delta MSE relative to A0 for the SAME:
        dataset
        seed
        experiment
        environment

    delta < 0 -> augmentation improved MSE
    delta > 0 -> augmentation harmed MSE
    """

    keys = [
        "dataset_id",
        "seed",
        "experiment",
        "environment",
    ]

    baseline = (
        df[df["augmentation_id"] == "A0"]
        [keys + ["mse"]]
        .rename(columns={"mse": "mse_A0"})
    )

    merged = df.merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    merged["delta_mse"] = (
        merged["mse"] - merged["mse_A0"]
    )

    assert not merged["mse_A0"].isna().any()

    # A0 must have exactly zero paired difference.
    a0 = merged[
        merged["augmentation_id"] == "A0"
    ]

    assert np.allclose(
        a0["delta_mse"],
        0.0,
    )

    return merged


def summarize_delta(delta):

    print("\n" + "=" * 70)
    print("2. PAIRED AUGMENTATION EFFECT")
    print("=" * 70)

    augmented = delta[
        delta["augmentation_id"] != "A0"
    ]

    summary = (
        augmented
        .groupby(
            ["dataset_id", "augmentation_id"],
            as_index=False
        )
        .agg(
            mean_delta_mse=("delta_mse", "mean"),
            median_delta_mse=("delta_mse", "median"),
            sd_delta_mse=("delta_mse", "std"),
            n=("delta_mse", "size"),
        )
    )

    summary["interpretation"] = np.where(
        summary["mean_delta_mse"] < 0,
        "lower MSE than A0",
        "higher MSE than A0",
    )

    print(summary.to_string(index=False))

    save_table(
        summary,
        "paired_delta_summary.csv"
    )


def severity_analysis(delta):

    print("\n" + "=" * 70)
    print("3. SEVERITY ANALYSIS")
    print("=" * 70)

    severity = delta[
        delta["experiment"] == "EXP1C"
    ].copy()

    summary = (
        severity
        .groupby(
            [
                "shift_type",
                "severity",
                "dataset_id",
                "augmentation_id",
            ],
            as_index=False,
        )
        .agg(
            mean_mse=("mse", "mean"),
            sd_mse=("mse", "std"),
            mean_delta_mse=("delta_mse", "mean"),
            sd_delta_mse=("delta_mse", "std"),
        )
    )

    save_table(
        summary,
        "severity_summary.csv"
    )

    return severity, summary


def compute_degradation(severity):

    print("\n" + "=" * 70)
    print("4. DEGRADATION FROM SEVERITY 0")
    print("=" * 70)

    keys = [
        "dataset_id",
        "augmentation_id",
        "seed",
        "shift_type",
    ]

    baseline = (
        severity[
            severity["severity"] == 0
        ][keys + ["mse"]]
        .rename(columns={"mse": "severity0_mse"})
    )

    degradation = severity.merge(
        baseline,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    degradation["degradation_mse"] = (
        degradation["mse"]
        - degradation["severity0_mse"]
    )

    assert not degradation[
        "severity0_mse"
    ].isna().any()

    save_table(
        degradation,
        "degradation.csv"
    )

    return degradation


def compute_auc(severity):
    """
    Compute AUC separately for each trained model and shift family.

    This preserves the repeated-measures structure.
    """

    print("\n" + "=" * 70)
    print("5. ROBUSTNESS AUC")
    print("=" * 70)

    rows = []

    groups = severity.groupby(
        [
            "model_id",
            "dataset_id",
            "augmentation_id",
            "seed",
            "shift_type",
        ]
    )

    for keys, group in groups:

        (
            model_id,
            dataset_id,
            augmentation_id,
            seed,
            shift_type,
        ) = keys

        group = group.sort_values(
            "shift_value"
        )

        x = group[
            "shift_value"
        ].to_numpy(dtype=float)

        y = group[
            "mse"
        ].to_numpy(dtype=float)

        auc = np.trapezoid(y, x)

        normalized_auc = (
            auc / (x.max() - x.min())
        )

        rows.append({
            "model_id": model_id,
            "dataset_id": dataset_id,
            "augmentation_id": augmentation_id,
            "seed": seed,
            "shift_type": shift_type,
            "auc": auc,
            "normalized_auc": normalized_auc,
        })

    auc_df = pd.DataFrame(rows)

    save_table(
        auc_df,
        "robustness_auc_per_model.csv"
    )

    summary = (
        auc_df
        .groupby(
            [
                "shift_type",
                "dataset_id",
                "augmentation_id",
            ],
            as_index=False,
        )
        .agg(
            mean_auc=("auc", "mean"),
            sd_auc=("auc", "std"),
            mean_normalized_auc=(
                "normalized_auc",
                "mean"
            ),
        )
    )

    save_table(
        summary,
        "robustness_auc_summary.csv"
    )

    return auc_df, summary


def plot_severity(summary):

    print("\n" + "=" * 70)
    print("6. PLOTS")
    print("=" * 70)

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        for dataset_id in [
            "D00",
            "D05",
            "D07",
        ]:

            subset = summary[
                (summary["shift_type"] == shift_type)
                &
                (summary["dataset_id"] == dataset_id)
            ]

            plt.figure(figsize=(8, 5))

            for augmentation in [
                "A0",
                "A1",
                "A2",
                "A3",
                "A4",
            ]:

                part = subset[
                    subset["augmentation_id"]
                    == augmentation
                ].sort_values("severity")

                plt.plot(
                    part["severity"],
                    part["mean_mse"],
                    marker="o",
                    label=augmentation,
                )

            plt.xlabel("Shift severity")
            plt.ylabel("Mean MSE")

            plt.title(
                f"{shift_type.capitalize()} shift - "
                f"{dataset_id}"
            )

            plt.legend()
            plt.tight_layout()

            path = OUTPUT / (
                f"{shift_type}_"
                f"{dataset_id}_severity.png"
            )

            plt.savefig(
                path,
                dpi=160,
            )

            plt.close()

            print("Saved:", path)


def engineering_report(
    df,
    delta,
    auc_summary,
):

    report = OUTPUT / "pilot_engineering_report.txt"

    augmented = delta[
        delta["augmentation_id"] != "A0"
    ]

    negative = (
        augmented["delta_mse"] < 0
    ).mean()

    positive = (
        augmented["delta_mse"] > 0
    ).mean()

    with open(
        report,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PHASE 2 ENGINEERING PILOT REPORT\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Evaluation records: {len(df)}\n"
        )

        f.write(
            f"Unique models: "
            f"{df['model_id'].nunique()}\n"
        )

        f.write(
            "Pilot purpose: engineering validation; "
            "not augmentation tuning.\n\n"
        )

        f.write(
            "PAIRED AUGMENTATION EFFECT\n"
        )

        f.write("-" * 60 + "\n")

        f.write(
            f"Fraction delta MSE < 0: "
            f"{negative:.4f}\n"
        )

        f.write(
            f"Fraction delta MSE > 0: "
            f"{positive:.4f}\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "Negative delta MSE means augmentation "
            "had lower MSE than paired A0.\n"
        )

        f.write(
            "Positive delta MSE means augmentation "
            "had higher MSE than paired A0.\n\n"
        )

        f.write(
            "These pilot results must NOT be used "
            "to tune A1-A4.\n"
        )

        f.write(
            "Exact winners are exploratory only.\n"
        )

        f.write(
            "Proceed to confirmation if data "
            "integrity and implementation diagnostics "
            "remain valid.\n"
        )

    print("Saved:", report)


def main():

    print("=" * 70)
    print("PHASE 2 PILOT ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT)

    integrity_check(df)

    delta = compute_paired_delta(df)

    save_table(
        delta,
        "paired_results.csv"
    )

    summarize_delta(delta)

    severity, severity_summary = (
        severity_analysis(delta)
    )

    degradation = compute_degradation(
        severity
    )

    auc_per_model, auc_summary = (
        compute_auc(severity)
    )

    plot_severity(
        severity_summary
    )

    engineering_report(
        df,
        delta,
        auc_summary,
    )

    print()
    print("=" * 70)
    print("PHASE 2 PILOT ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()