from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "experiment_5"
    / "confirmation"
    / "analysis"
)

OUTPUT_FILE = (
    ANALYSIS_DIR
    / "phase5_h55_permutation_diagnostic.csv"
)


# Observed F statistics from the frozen H5.5 analysis.
# The script will also try to read them from the summary file.

FALLBACK_OBSERVED = {
    "temperature": 0.941,
    "humidity": 1.713,
}


def load_observed():

    summary_path = (
        ANALYSIS_DIR
        / "phase5_h55_permutation_summary.csv"
    )

    if not summary_path.exists():
        return FALLBACK_OBSERVED

    summary = pd.read_csv(summary_path)

    observed = {}

    for _, row in summary.iterrows():

        observed[
            row["shift_type"]
        ] = float(
            row["observed_f"]
        )

    return observed


def diagnose_family(
    shift_type,
    observed_f,
):

    path = (
        ANALYSIS_DIR
        / f"phase5_h55_permutation_{shift_type}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "f_statistic" not in df.columns:
        raise RuntimeError(
            f"{path.name}: missing f_statistic column."
        )

    values = (
        df["f_statistic"]
        .astype(float)
        .to_numpy()
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{shift_type}: non-finite permutation statistics."
        )

    n = len(values)

    n_ge = int(
        np.sum(
            values >= observed_f
        )
    )

    empirical_p = (
        1 + n_ge
    ) / (
        n + 1
    )

    return {
        "shift_type": shift_type,
        "observed_f": observed_f,
        "n_permutations": n,

        "perm_mean": np.mean(values),
        "perm_sd": np.std(values, ddof=1),

        "perm_min": np.min(values),

        "perm_q01": np.quantile(
            values, 0.01
        ),

        "perm_q05": np.quantile(
            values, 0.05
        ),

        "perm_q25": np.quantile(
            values, 0.25
        ),

        "perm_median": np.quantile(
            values, 0.50
        ),

        "perm_q75": np.quantile(
            values, 0.75
        ),

        "perm_q95": np.quantile(
            values, 0.95
        ),

        "perm_q99": np.quantile(
            values, 0.99
        ),

        "perm_max": np.max(values),

        "n_perm_ge_observed": n_ge,

        "fraction_ge_observed": (
            n_ge / n
        ),

        "empirical_p_plus1": empirical_p,

        "observed_percentile": (
            100.0
            * np.mean(
                values < observed_f
            )
        ),
    }


def main():

    print("=" * 76)
    print("PHASE 5 H5.5 PERMUTATION DIAGNOSTIC")
    print("=" * 76)

    observed = load_observed()

    results = []

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        if shift_type not in observed:
            raise RuntimeError(
                f"Missing observed F for {shift_type}"
            )

        result = diagnose_family(
            shift_type,
            observed[shift_type],
        )

        results.append(result)

    diagnostic = pd.DataFrame(
        results
    )

    diagnostic.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        diagnostic.to_string(
            index=False
        )
    )

    print()
    print("=" * 76)
    print("INTERPRETATION CHECK")
    print("=" * 76)

    for _, row in diagnostic.iterrows():

        print()
        print(
            row["shift_type"].upper()
        )

        print(
            f"Observed F: "
            f"{row['observed_f']:.6f}"
        )

        print(
            f"Permutation mean: "
            f"{row['perm_mean']:.6f}"
        )

        print(
            f"Permutation median: "
            f"{row['perm_median']:.6f}"
        )

        print(
            f"95th percentile: "
            f"{row['perm_q95']:.6f}"
        )

        print(
            f"99th percentile: "
            f"{row['perm_q99']:.6f}"
        )

        print(
            f"Observed percentile: "
            f"{row['observed_percentile']:.2f}%"
        )

        print(
            f"Permutations >= observed: "
            f"{int(row['n_perm_ge_observed'])}"
        )

        print(
            f"Empirical p (+1): "
            f"{row['empirical_p_plus1']:.6f}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )

    print()
    print("=" * 76)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()