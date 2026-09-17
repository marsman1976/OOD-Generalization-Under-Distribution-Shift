from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PHASE 4 — POST-HOC DIAGNOSTIC ANALYSIS
# ============================================================
#
# IMPORTANT:
#
# This analysis is POST-HOC / DIAGNOSTIC.
#
# It does NOT replace or modify the preregistered primary
# Phase 4 hypothesis H4.5.
#
# Primary Phase 4 conclusion remains:
#
#   The seed-respecting omnibus A × M × V test was
#   not significant for temperature or humidity.
#
#
# Diagnostic questions:
#
# D1:
# Why did the temperature component
#
#   A4 vs A1 × M2 vs M0 × severity
#
# show a corrected component-level signal?
#
#
# D2:
# Is that signal broadly present across seeds or driven
# by only one/few seeds?
#
#
# D3:
# Why does per-severity A × M evidence weaken at the
# most severe temperature/humidity shifts?
#
#
# No hypothesis redesign is allowed from this script.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "experiment_4"
    / "confirmation"
    / "analysis"
)

FORMAL_DIR = (
    ANALYSIS_DIR
    / "formal_interactions"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "posthoc_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PAIRED_FILE = (
    ANALYSIS_DIR
    / "paired_augmentation_effects.csv"
)

CONTRAST_FILE = (
    FORMAL_DIR
    / "H45_seed_level_contrasts.csv"
)


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

AUGMENTATIONS = [
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


# ============================================================
# Load
# ============================================================

def load_data():

    if not PAIRED_FILE.exists():
        raise FileNotFoundError(PAIRED_FILE)

    df = pd.read_csv(PAIRED_FILE)

    df = df[
        df["experiment"] == "EXP1C"
    ].copy()

    assert len(df) == 1560
    assert set(df["seed"].unique()) == set(SEEDS)

    assert np.isfinite(
        df["delta_mse"].to_numpy()
    ).all()

    return df


# ============================================================
# Diagnostic 1:
#
# Extract localized temperature component:
#
# A4 vs A1 × M2 vs M0
#
# at every severity and seed.
#
# Difference-in-differences:
#
# [A4,M2 - A4,M0]
# -
# [A1,M2 - A1,M0]
#
# ============================================================

def localized_temperature_component(df):

    temp = df[
        df["shift_type"] == "temperature"
    ].copy()

    records = []

    for seed in SEEDS:

        seed_df = temp[
            temp["seed"] == seed
        ]

        for severity in sorted(
            seed_df["severity"].unique()
        ):

            level = seed_df[
                seed_df["severity"] == severity
            ]

            def value(a, m):

                row = level[
                    (
                        level["augmentation_id"] == a
                    )
                    &
                    (
                        level["model_config"] == m
                    )
                ]

                if len(row) != 1:
                    raise RuntimeError(
                        f"Expected one row: "
                        f"seed={seed}, severity={severity}, "
                        f"A={a}, M={m}; got {len(row)}"
                    )

                return float(
                    row["delta_mse"].iloc[0]
                )

            a4_m2 = value("A4", "M2")
            a4_m0 = value("A4", "M0")

            a1_m2 = value("A1", "M2")
            a1_m0 = value("A1", "M0")

            did = (
                (a4_m2 - a4_m0)
                -
                (a1_m2 - a1_m0)
            )

            shift_value = float(
                level["shift_value"].iloc[0]
            )

            records.append({
                "seed": seed,
                "severity": int(severity),
                "shift_value": shift_value,
                "A4_M2": a4_m2,
                "A4_M0": a4_m0,
                "A1_M2": a1_m2,
                "A1_M0": a1_m0,
                "difference_in_difference": did,
            })

    result = pd.DataFrame(records)

    assert len(result) == 60

    return result


# ============================================================
# Summarize localized component by severity
# ============================================================

def summarize_localized_component(component_df):

    summary = (
        component_df
        .groupby(
            [
                "severity",
                "shift_value",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=(
                "seed",
                "nunique",
            ),
            mean_did=(
                "difference_in_difference",
                "mean",
            ),
            median_did=(
                "difference_in_difference",
                "median",
            ),
            sd_did=(
                "difference_in_difference",
                "std",
            ),
            min_did=(
                "difference_in_difference",
                "min",
            ),
            max_did=(
                "difference_in_difference",
                "max",
            ),
            fraction_negative=(
                "difference_in_difference",
                lambda x: np.mean(x < 0),
            ),
            fraction_positive=(
                "difference_in_difference",
                lambda x: np.mean(x > 0),
            ),
        )
    )

    return summary


# ============================================================
# Seed-level slope of localized component
# ============================================================

def seed_component_slopes(component_df):

    records = []

    for seed in SEEDS:

        part = (
            component_df[
                component_df["seed"] == seed
            ]
            .sort_values("severity")
        )

        x = part[
            "severity"
        ].to_numpy(dtype=float)

        y = part[
            "difference_in_difference"
        ].to_numpy(dtype=float)

        slope = np.polyfit(
            x,
            y,
            1,
        )[0]

        records.append({
            "seed": seed,
            "slope": float(slope),
            "negative_slope": bool(slope < 0),
        })

    return pd.DataFrame(records)


# ============================================================
# Leave-one-seed-out diagnostic
#
# Does the localized slope disappear when one seed is removed?
# ============================================================

def leave_one_seed_out(component_df):

    records = []

    for dropped_seed in SEEDS:

        part = component_df[
            component_df["seed"]
            != dropped_seed
        ]

        seed_slopes = []

        for seed in sorted(
            part["seed"].unique()
        ):

            seed_part = (
                part[
                    part["seed"] == seed
                ]
                .sort_values("severity")
            )

            slope = np.polyfit(
                seed_part["severity"],
                seed_part["difference_in_difference"],
                1,
            )[0]

            seed_slopes.append(
                float(slope)
            )

        records.append({
            "dropped_seed": dropped_seed,
            "n_remaining_seeds": len(seed_slopes),
            "mean_slope": float(
                np.mean(seed_slopes)
            ),
            "median_slope": float(
                np.median(seed_slopes)
            ),
            "sd_slope": float(
                np.std(seed_slopes, ddof=1)
            ),
            "fraction_negative": float(
                np.mean(
                    np.asarray(seed_slopes) < 0
                )
            ),
        })

    return pd.DataFrame(records)


# ============================================================
# Diagnostic 2:
#
# Examine variance at each severity.
#
# We want to know whether weaker A × M evidence at severe
# shifts coincides with greater across-seed variability.
# ============================================================

def severity_variability(df):

    records = []

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        shift_df = df[
            df["shift_type"] == shift_type
        ]

        for severity in sorted(
            shift_df["severity"].unique()
        ):

            level = shift_df[
                shift_df["severity"] == severity
            ]

            shift_value = float(
                level["shift_value"].iloc[0]
            )

            # Across all A × M × seed delta values
            overall_mean = float(
                level["delta_mse"].mean()
            )

            overall_sd = float(
                level["delta_mse"].std(ddof=1)
            )

            overall_median = float(
                level["delta_mse"].median()
            )

            records.append({
                "shift_type": shift_type,
                "severity": int(severity),
                "shift_value": shift_value,
                "n": len(level),
                "mean_delta_mse": overall_mean,
                "median_delta_mse": overall_median,
                "sd_delta_mse": overall_sd,
                "variance_delta_mse": overall_sd ** 2,
            })

    return pd.DataFrame(records)


# ============================================================
# Cell-level variability:
#
# A × M × severity
# ============================================================

def cell_variability(df):

    result = (
        df
        .groupby(
            [
                "shift_type",
                "severity",
                "shift_value",
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
            min_delta_mse=(
                "delta_mse",
                "min",
            ),
            max_delta_mse=(
                "delta_mse",
                "max",
            ),
            fraction_improved=(
                "delta_mse",
                lambda x: np.mean(x < 0),
            ),
        )
    )

    return result


# ============================================================
# Plot 1:
# localized component by seed
# ============================================================

def plot_localized_component(component_df):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for seed in SEEDS:

        part = (
            component_df[
                component_df["seed"] == seed
            ]
            .sort_values("shift_value")
        )

        ax.plot(
            part["shift_value"],
            part["difference_in_difference"],
            marker="o",
            alpha=0.55,
            label=f"Seed {seed}",
        )

    mean_curve = (
        component_df
        .groupby(
            "shift_value",
            as_index=False,
        )[
            "difference_in_difference"
        ]
        .mean()
    )

    ax.plot(
        mean_curve["shift_value"],
        mean_curve["difference_in_difference"],
        marker="o",
        linewidth=3,
        label="Mean",
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Temperature shift value"
    )

    ax.set_ylabel(
        "A4-vs-A1 × M2-vs-M0 interaction contrast"
    )

    ax.set_title(
        "Phase 4 post-hoc diagnostic:\n"
        "localized temperature interaction by seed"
    )

    ax.legend(
        fontsize=7,
        ncol=2,
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "localized_temperature_component_by_seed.png"
    )

    fig.savefig(
        path,
        dpi=180,
    )

    plt.close(fig)


# ============================================================
# Plot 2:
# variability versus severity
# ============================================================

def plot_variability(variability_df):

    for shift_type in [
        "temperature",
        "humidity",
    ]:

        part = (
            variability_df[
                variability_df["shift_type"]
                == shift_type
            ]
            .sort_values("shift_value")
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            part["shift_value"],
            part["sd_delta_mse"],
            marker="o",
        )

        ax.set_xlabel(
            f"{shift_type.capitalize()} shift value"
        )

        ax.set_ylabel(
            "SD of paired ΔMSE"
        )

        ax.set_title(
            f"Phase 4 post-hoc diagnostic:\n"
            f"ΔMSE variability across "
            f"{shift_type} severity"
        )

        fig.tight_layout()

        path = (
            OUTPUT_DIR
            / f"{shift_type}_severity_variability.png"
        )

        fig.savefig(
            path,
            dpi=180,
        )

        plt.close(fig)


# ============================================================
# Optional consistency check against formal contrast file
# ============================================================

def check_formal_contrast(
    slope_df
):

    if not CONTRAST_FILE.exists():

        return {
            "formal_contrast_file_found": False,
        }

    formal = pd.read_csv(
        CONTRAST_FILE
    )

    formal = formal[
        formal["shift_type"]
        == "temperature"
    ].copy()

    target_column = (
        "A4_vs_A1__M2_vs_M0"
    )

    if target_column not in formal.columns:

        return {
            "formal_contrast_file_found": True,
            "target_column_found": False,
        }

    merged = slope_df.merge(
        formal[
            [
                "seed",
                target_column,
            ]
        ],
        on="seed",
        how="inner",
    )

    # Our diagnostic slope uses raw integer severity.
    # Formal analysis used standardized severity.
    #
    # Therefore magnitude differs by a constant scaling
    # factor, but direction should match.

    same_direction = (
        np.sign(
            merged["slope"]
        )
        ==
        np.sign(
            merged[target_column]
        )
    )

    return {
        "formal_contrast_file_found": True,
        "target_column_found": True,
        "n_matched_seeds": int(len(merged)),
        "same_direction_all_seeds": bool(
            same_direction.all()
        ),
        "same_direction_fraction": float(
            same_direction.mean()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 78)
    print(
        "PHASE 4 POST-HOC DIAGNOSTIC ANALYSIS"
    )
    print("=" * 78)

    print()
    print(
        "IMPORTANT: This is diagnostic only."
    )

    print(
        "It does NOT change the primary H4.5 conclusion."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Localized temperature component
    # --------------------------------------------------------

    component = (
        localized_temperature_component(
            df
        )
    )

    component.to_csv(
        OUTPUT_DIR
        / "localized_temperature_component.csv",
        index=False,
    )

    component_summary = (
        summarize_localized_component(
            component
        )
    )

    component_summary.to_csv(
        OUTPUT_DIR
        / "localized_temperature_component_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Seed slopes
    # --------------------------------------------------------

    slopes = seed_component_slopes(
        component
    )

    slopes.to_csv(
        OUTPUT_DIR
        / "localized_component_seed_slopes.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Leave-one-seed-out
    # --------------------------------------------------------

    loso = leave_one_seed_out(
        component
    )

    loso.to_csv(
        OUTPUT_DIR
        / "localized_component_leave_one_seed_out.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Severity variability
    # --------------------------------------------------------

    variability = severity_variability(
        df
    )

    variability.to_csv(
        OUTPUT_DIR
        / "severity_variability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Cell variability
    # --------------------------------------------------------

    cells = cell_variability(
        df
    )

    cells.to_csv(
        OUTPUT_DIR
        / "cell_variability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_localized_component(
        component
    )

    plot_variability(
        variability
    )

    # --------------------------------------------------------
    # Consistency check
    # --------------------------------------------------------

    consistency = check_formal_contrast(
        slopes
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "phase": "PHASE4",
        "analysis": "POSTHOC_DIAGNOSTIC",

        "status": (
            "Diagnostic only; does not modify "
            "preregistered H4.5."
        ),

        "diagnostic_target": (
            "Temperature A4-vs-A1 x "
            "M2-vs-M0 x severity component"
        ),

        "n_seeds": 10,

        "localized_component": {
            "mean_seed_slope":
                float(
                    slopes["slope"].mean()
                ),

            "median_seed_slope":
                float(
                    slopes["slope"].median()
                ),

            "sd_seed_slope":
                float(
                    slopes["slope"].std(ddof=1)
                ),

            "fraction_negative_slopes":
                float(
                    np.mean(
                        slopes["slope"] < 0
                    )
                ),

            "fraction_positive_slopes":
                float(
                    np.mean(
                        slopes["slope"] > 0
                    )
                ),
        },

        "formal_consistency_check":
            consistency,

        "interpretation_rules": [
            (
                "Do not reinterpret this diagnostic "
                "as the primary Phase 4 result."
            ),
            (
                "Do not claim a universal A x M x V "
                "interaction from a localized component."
            ),
            (
                "Leave-one-seed-out results are used "
                "to assess concentration of the signal."
            ),
            (
                "Increasing variance at severe shifts "
                "is a possible explanation only if "
                "supported by the diagnostic output."
            ),
            (
                "Any new hypothesis generated here "
                "belongs to future work."
            ),
        ],
    }

    with open(
        OUTPUT_DIR
        / "phase4_posthoc_diagnostic_report.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "LOCALIZED TEMPERATURE COMPONENT"
    )
    print("=" * 78)
    print()

    print(
        component_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "SEED-LEVEL SLOPES"
    )
    print("=" * 78)
    print()

    print(
        slopes.to_string(
            index=False
        )
    )

    print()
    print(
        "Mean slope:",
        slopes["slope"].mean()
    )

    print(
        "Negative slopes:",
        int(
            (
                slopes["slope"] < 0
            ).sum()
        ),
        "/ 10"
    )

    print()
    print("=" * 78)
    print(
        "LEAVE-ONE-SEED-OUT"
    )
    print("=" * 78)
    print()

    print(
        loso.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "SEVERITY VARIABILITY"
    )
    print("=" * 78)
    print()

    print(
        variability.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "FORMAL-CONTRAST CONSISTENCY"
    )
    print("=" * 78)
    print()

    print(
        json.dumps(
            consistency,
            indent=2,
        )
    )

    print()
    print("=" * 78)
    print(
        "PHASE 4 POST-HOC DIAGNOSTIC: COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "Results saved in:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()