"""
Phase 5 post-confirmation effect-pattern analysis.

Purpose
-------
Explain the already-frozen Phase 5 confirmation results.

This script:
    - performs NO model training
    - performs NO new confirmatory hypothesis testing
    - does NOT modify H5.5 or H5.6
    - does NOT select a universal "best" configuration

It summarizes:
    1. D x A x M trajectory-level augmentation effects
    2. D x A x M effects at each shift severity
    3. seed-level directional stability
    4. concentration/dispersion of observed effects
    5. severity-dependent effect trajectories
    6. paper-ready descriptive tables and figures

Negative delta values mean augmentation reduced error relative to A0.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CONFIRMATION_DIR = (
    ROOT
    / "results"
    / "experiment_5"
    / "confirmation"
)

ANALYSIS_DIR = CONFIRMATION_DIR / "analysis"

OUTPUT_DIR = ANALYSIS_DIR / "effect_patterns"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DELTA_AUC_FILE = (
    ANALYSIS_DIR
    / "phase5_delta_auc.csv"
)

POINTWISE_FILE = (
    ANALYSIS_DIR
    / "phase5_pointwise_delta.csv"
)


# ============================================================
# Frozen design
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

SEEDS = [
    121, 132, 143, 154, 165,
    176, 187, 198, 209, 220,
]

SHIFT_TYPES = [
    "temperature",
    "humidity",
]


# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def require(condition, message):
    if not condition:
        raise RuntimeError(
            f"INTEGRITY ERROR: {message}"
        )


def ci95(series):
    """
    Descriptive normal-approximation 95% CI for the mean.

    Used only for descriptive uncertainty visualization.
    It is NOT a replacement for the frozen confirmatory tests.
    """

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    n = len(x)

    if n < 2:
        return np.nan, np.nan

    mean = x.mean()

    se = (
        x.std(ddof=1)
        / np.sqrt(n)
    )

    margin = 1.96 * se

    return (
        mean - margin,
        mean + margin,
    )


# ============================================================
# Load
# ============================================================

print_section(
    "PHASE 5 EFFECT-PATTERN ANALYSIS"
)

print(
    f"Loading:\n"
    f"  {DELTA_AUC_FILE}\n"
    f"  {POINTWISE_FILE}"
)

require(
    DELTA_AUC_FILE.exists(),
    "phase5_delta_auc.csv not found",
)

require(
    POINTWISE_FILE.exists(),
    "phase5_pointwise_delta.csv not found",
)


delta_auc = pd.read_csv(
    DELTA_AUC_FILE
)

pointwise = pd.read_csv(
    POINTWISE_FILE
)


# ============================================================
# Integrity checks
# ============================================================

print_section(
    "1. INPUT INTEGRITY"
)


require(
    len(delta_auc) == 2400,
    f"Expected 2400 delta-AUC rows, got {len(delta_auc)}",
)

require(
    len(pointwise) == 15600,
    f"Expected 15600 pointwise rows, got {len(pointwise)}",
)


for df, name in [
    (delta_auc, "delta_auc"),
    (pointwise, "pointwise"),
]:

    require(
        set(df["dataset_id"].unique())
        == set(DATASETS),
        f"{name}: incorrect dataset levels",
    )

    require(
        set(df["augmentation_id"].unique())
        == set(AUGMENTATIONS),
        f"{name}: incorrect augmentation levels",
    )

    require(
        set(df["model_config"].unique())
        == set(MODELS),
        f"{name}: incorrect model levels",
    )

    require(
        set(
            df["seed"]
            .astype(int)
            .unique()
        )
        == set(SEEDS),
        f"{name}: incorrect seed levels",
    )

    require(
        set(df["shift_type"].unique())
        == set(SHIFT_TYPES),
        f"{name}: incorrect shift families",
    )


require(
    np.isfinite(
        delta_auc[
            "delta_auc_mse"
        ].to_numpy(dtype=float)
    ).all(),
    "Non-finite delta-AUC values",
)


require(
    np.isfinite(
        pointwise[
            "delta_mse"
        ].to_numpy(dtype=float)
    ).all(),
    "Non-finite pointwise delta-MSE values",
)


print(
    "[PASS] Input integrity verified."
)


# ============================================================
# 2. Trajectory-level D x A x M effects
# ============================================================

print_section(
    "2. TRAJECTORY-LEVEL D x A x M EFFECTS"
)


trajectory_rows = []


group_cols = [
    "shift_type",
    "dataset_id",
    "augmentation_id",
    "model_config",
]


for keys, g in delta_auc.groupby(
    group_cols,
    observed=True,
):

    shift_type, dataset_id, augmentation_id, model_config = keys

    values = (
        g["delta_auc_mse"]
        .to_numpy(dtype=float)
    )

    low, high = ci95(
        g["delta_auc_mse"]
    )

    trajectory_rows.append({

        "shift_type":
            shift_type,

        "dataset_id":
            dataset_id,

        "augmentation_id":
            augmentation_id,

        "model_config":
            model_config,

        "n_seeds":
            len(g),

        "mean_delta_auc_mse":
            np.mean(values),

        "median_delta_auc_mse":
            np.median(values),

        "sd_delta_auc_mse":
            np.std(
                values,
                ddof=1,
            ),

        "ci95_low":
            low,

        "ci95_high":
            high,

        "min_delta_auc_mse":
            np.min(values),

        "max_delta_auc_mse":
            np.max(values),

        "fraction_improved":
            np.mean(
                values < 0
            ),

        "fraction_harmed":
            np.mean(
                values > 0
            ),

        "all_seeds_improved":
            bool(
                np.all(
                    values < 0
                )
            ),

        "all_seeds_harmed":
            bool(
                np.all(
                    values > 0
                )
            ),
    })


trajectory = pd.DataFrame(
    trajectory_rows
)


require(
    len(trajectory) == 240,
    (
        "Expected 240 trajectory cells "
        f"but got {len(trajectory)}"
    ),
)


trajectory = trajectory.sort_values(
    [
        "shift_type",
        "dataset_id",
        "augmentation_id",
        "model_config",
    ]
)


trajectory.to_csv(
    OUTPUT_DIR
    / "trajectory_cell_effects.csv",
    index=False,
)


print(
    f"Trajectory cells: {len(trajectory)}"
)


# ============================================================
# 3. Pointwise severity D x A x M effects
# ============================================================

print_section(
    "3. SEVERITY-LEVEL D x A x M EFFECTS"
)


severity_rows = []


severity_group_cols = [
    "shift_type",
    "severity",
    "shift_value",
    "dataset_id",
    "augmentation_id",
    "model_config",
]


for keys, g in pointwise.groupby(
    severity_group_cols,
    observed=True,
):

    (
        shift_type,
        severity,
        shift_value,
        dataset_id,
        augmentation_id,
        model_config,
    ) = keys

    values = (
        g["delta_mse"]
        .to_numpy(dtype=float)
    )

    low, high = ci95(
        g["delta_mse"]
    )

    severity_rows.append({

        "shift_type":
            shift_type,

        "severity":
            int(severity),

        "shift_value":
            float(shift_value),

        "dataset_id":
            dataset_id,

        "augmentation_id":
            augmentation_id,

        "model_config":
            model_config,

        "n_seeds":
            len(g),

        "mean_delta_mse":
            np.mean(values),

        "median_delta_mse":
            np.median(values),

        "sd_delta_mse":
            np.std(
                values,
                ddof=1,
            ),

        "ci95_low":
            low,

        "ci95_high":
            high,

        "min_delta_mse":
            np.min(values),

        "max_delta_mse":
            np.max(values),

        "fraction_improved":
            np.mean(
                values < 0
            ),

        "fraction_harmed":
            np.mean(
                values > 0
            ),

        "all_seeds_improved":
            bool(
                np.all(
                    values < 0
                )
            ),

        "all_seeds_harmed":
            bool(
                np.all(
                    values > 0
                )
            ),
    })


severity_cells = pd.DataFrame(
    severity_rows
)


expected_severity_cells = (
    (6 + 7)
    * 10
    * 4
    * 3
)


require(
    len(severity_cells)
    == expected_severity_cells,
    (
        "Expected "
        f"{expected_severity_cells} severity cells, "
        f"got {len(severity_cells)}"
    ),
)


severity_cells = severity_cells.sort_values(
    [
        "shift_type",
        "severity",
        "dataset_id",
        "augmentation_id",
        "model_config",
    ]
)


severity_cells.to_csv(
    OUTPUT_DIR
    / "severity_cell_effects.csv",
    index=False,
)


print(
    f"Severity cells: {len(severity_cells)}"
)


# ============================================================
# 4. Seed stability
# ============================================================

print_section(
    "4. SEED STABILITY"
)


seed_stability = trajectory[
    [
        "shift_type",
        "dataset_id",
        "augmentation_id",
        "model_config",
        "n_seeds",
        "mean_delta_auc_mse",
        "median_delta_auc_mse",
        "sd_delta_auc_mse",
        "fraction_improved",
        "fraction_harmed",
        "all_seeds_improved",
        "all_seeds_harmed",
    ]
].copy()


seed_stability["direction"] = np.select(

    [
        seed_stability[
            "fraction_improved"
        ] >= 0.8,

        seed_stability[
            "fraction_harmed"
        ] >= 0.8,
    ],

    [
        "mostly_improved",
        "mostly_harmed",
    ],

    default="mixed",
)


seed_stability.to_csv(
    OUTPUT_DIR
    / "seed_stability.csv",
    index=False,
)


print(
    seed_stability[
        "direction"
    ].value_counts()
)


# ============================================================
# 5. Interaction concentration / dispersion
# ============================================================

print_section(
    "5. INTERACTION CONCENTRATION"
)


concentration_rows = []


for (
    shift_type,
    severity,
    shift_value
), g in severity_cells.groupby(
    [
        "shift_type",
        "severity",
        "shift_value",
    ],
    observed=True,
):

    effects = (
        g["mean_delta_mse"]
        .to_numpy(dtype=float)
    )

    abs_effects = np.abs(
        effects
    )

    total_abs = abs_effects.sum()

    if total_abs > 0:

        shares = (
            np.sort(abs_effects)[::-1]
            / total_abs
        )

        top1_share = shares[0]

        top5_share = (
            shares[
                :min(
                    5,
                    len(shares),
                )
            ].sum()
        )

        top10_share = (
            shares[
                :min(
                    10,
                    len(shares),
                )
            ].sum()
        )

    else:

        top1_share = np.nan
        top5_share = np.nan
        top10_share = np.nan

    concentration_rows.append({

        "shift_type":
            shift_type,

        "severity":
            int(severity),

        "shift_value":
            float(shift_value),

        "n_cells":
            len(g),

        "mean_cell_effect":
            np.mean(effects),

        "sd_across_cells":
            np.std(
                effects,
                ddof=1,
            ),

        "min_cell_effect":
            np.min(effects),

        "max_cell_effect":
            np.max(effects),

        "range_cell_effect":
            (
                np.max(effects)
                - np.min(effects)
            ),

        "fraction_cells_improved":
            np.mean(
                effects < 0
            ),

        "fraction_cells_harmed":
            np.mean(
                effects > 0
            ),

        "top1_absolute_effect_share":
            top1_share,

        "top5_absolute_effect_share":
            top5_share,

        "top10_absolute_effect_share":
            top10_share,
    })


concentration = pd.DataFrame(
    concentration_rows
)


require(
    len(concentration) == 13,
    "Expected 13 concentration rows",
)


concentration.to_csv(
    OUTPUT_DIR
    / "interaction_concentration.csv",
    index=False,
)


# ============================================================
# 6. Observed extremes
# ============================================================

print_section(
    "6. OBSERVED EFFECT EXTREMES"
)


extreme_rows = []


for shift_type in SHIFT_TYPES:

    g = trajectory.loc[
        trajectory["shift_type"]
        == shift_type
    ].copy()

    most_negative = (
        g.nsmallest(
            10,
            "mean_delta_auc_mse",
        )
    )

    most_positive = (
        g.nlargest(
            10,
            "mean_delta_auc_mse",
        )
    )

    for rank, (_, row) in enumerate(
        most_negative.iterrows(),
        start=1,
    ):

        extreme_rows.append({

            "shift_type":
                shift_type,

            "category":
                "largest_observed_reduction",

            "rank":
                rank,

            "dataset_id":
                row["dataset_id"],

            "augmentation_id":
                row["augmentation_id"],

            "model_config":
                row["model_config"],

            "mean_delta_auc_mse":
                row["mean_delta_auc_mse"],

            "fraction_improved":
                row["fraction_improved"],
        })

    for rank, (_, row) in enumerate(
        most_positive.iterrows(),
        start=1,
    ):

        extreme_rows.append({

            "shift_type":
                shift_type,

            "category":
                "largest_observed_increase",

            "rank":
                rank,

            "dataset_id":
                row["dataset_id"],

            "augmentation_id":
                row["augmentation_id"],

            "model_config":
                row["model_config"],

            "mean_delta_auc_mse":
                row["mean_delta_auc_mse"],

            "fraction_improved":
                row["fraction_improved"],
        })


extremes = pd.DataFrame(
    extreme_rows
)


extremes.to_csv(
    OUTPUT_DIR
    / "observed_effect_extremes.csv",
    index=False,
)


print(
    "Observed extremes exported."
)

print(
    "These are descriptive observations, "
    "not universal recommendations."
)


# ============================================================
# 7. Overall augmentation x model trajectories
# ============================================================

print_section(
    "7. A x M EFFECT TRAJECTORIES"
)


am_rows = []


for keys, g in pointwise.groupby(
    [
        "shift_type",
        "severity",
        "shift_value",
        "augmentation_id",
        "model_config",
    ],
    observed=True,
):

    (
        shift_type,
        severity,
        shift_value,
        augmentation_id,
        model_config,
    ) = keys

    values = (
        g["delta_mse"]
        .to_numpy(dtype=float)
    )

    low, high = ci95(
        g["delta_mse"]
    )

    am_rows.append({

        "shift_type":
            shift_type,

        "severity":
            int(severity),

        "shift_value":
            float(shift_value),

        "augmentation_id":
            augmentation_id,

        "model_config":
            model_config,

        "n":
            len(g),

        "mean_delta_mse":
            np.mean(values),

        "median_delta_mse":
            np.median(values),

        "sd_delta_mse":
            np.std(
                values,
                ddof=1,
            ),

        "ci95_low":
            low,

        "ci95_high":
            high,

        "fraction_improved":
            np.mean(
                values < 0
            ),
    })


am_trajectory = pd.DataFrame(
    am_rows
)


require(
    len(am_trajectory)
    == (
        13
        * 4
        * 3
    ),
    "Incorrect A x M trajectory table size",
)


am_trajectory.to_csv(
    OUTPUT_DIR
    / "augmentation_model_trajectories.csv",
    index=False,
)


# ============================================================
# 8. Plot A x M mean effect trajectories
# ============================================================

print_section(
    "8. GENERATING FIGURES"
)


for shift_type in SHIFT_TYPES:

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    subset = am_trajectory.loc[
        am_trajectory[
            "shift_type"
        ] == shift_type
    ].copy()

    for augmentation_id in AUGMENTATIONS:

        for model_config in MODELS:

            g = subset.loc[
                (
                    subset[
                        "augmentation_id"
                    ] == augmentation_id
                )
                &
                (
                    subset[
                        "model_config"
                    ] == model_config
                )
            ].sort_values(
                "shift_value"
            )

            label = (
                f"{augmentation_id}-{model_config}"
            )

            ax.plot(
                g["shift_value"],
                g["mean_delta_mse"],
                marker="o",
                label=label,
            )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        (
            "Temperature (°C)"
            if shift_type == "temperature"
            else "Humidity (%)"
        )
    )

    ax.set_ylabel(
        "Mean ΔMSE relative to A0"
    )

    ax.set_title(
        (
            "Phase 5: augmentation × model "
            f"effect trajectories — {shift_type}"
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

    output_path = (
        OUTPUT_DIR
        / f"{shift_type}_effect_trajectories.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path.name}"
    )


# ============================================================
# 9. Plot interaction dispersion by severity
# ============================================================

for shift_type in SHIFT_TYPES:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    g = concentration.loc[
        concentration[
            "shift_type"
        ] == shift_type
    ].sort_values(
        "shift_value"
    )

    ax.plot(
        g["shift_value"],
        g["sd_across_cells"],
        marker="o",
    )

    ax.set_xlabel(
        (
            "Temperature (°C)"
            if shift_type == "temperature"
            else "Humidity (%)"
        )
    )

    ax.set_ylabel(
        "SD of mean ΔMSE across D×A×M cells"
    )

    ax.set_title(
        (
            "Phase 5: cross-cell effect dispersion "
            f"— {shift_type}"
        )
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{shift_type}_effect_dispersion.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 10. Compact paper table
# ============================================================

print_section(
    "9. PAPER-READY SUMMARY TABLE"
)


paper_rows = []


for shift_type in SHIFT_TYPES:

    g = trajectory.loc[
        trajectory[
            "shift_type"
        ] == shift_type
    ]

    paper_rows.append({

        "shift_type":
            shift_type,

        "n_DxAxM_cells":
            len(g),

        "mean_delta_auc_mse":
            g[
                "mean_delta_auc_mse"
            ].mean(),

        "median_cell_delta_auc_mse":
            g[
                "mean_delta_auc_mse"
            ].median(),

        "fraction_cells_mean_improved":
            (
                g[
                    "mean_delta_auc_mse"
                ] < 0
            ).mean(),

        "fraction_cells_80pct_seed_improved":
            (
                g[
                    "fraction_improved"
                ] >= 0.8
            ).mean(),

        "fraction_cells_all_seeds_improved":
            g[
                "all_seeds_improved"
            ].mean(),

        "cross_cell_sd":
            g[
                "mean_delta_auc_mse"
            ].std(ddof=1),

        "min_observed_cell_mean":
            g[
                "mean_delta_auc_mse"
            ].min(),

        "max_observed_cell_mean":
            g[
                "mean_delta_auc_mse"
            ].max(),
    })


paper_summary = pd.DataFrame(
    paper_rows
)


paper_summary.to_csv(
    OUTPUT_DIR
    / "paper_effect_summary.csv",
    index=False,
)


print(
    paper_summary.to_string(
        index=False
    )
)


# ============================================================
# 11. Text report
# ============================================================

print_section(
    "10. WRITING EFFECT-PATTERN REPORT"
)


report_lines = []

report_lines.append(
    "PHASE 5 POST-CONFIRMATION EFFECT-PATTERN REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "STATUS: DESCRIPTIVE / POST-CONFIRMATION"
)

report_lines.append("")

report_lines.append(
    "This analysis explains the already-frozen Phase 5 "
    "confirmation results. It introduces no new confirmatory "
    "hypotheses and does not alter H5.5 or H5.6."
)

report_lines.append("")

report_lines.append(
    "Negative delta values indicate lower error under the "
    "augmentation relative to A0."
)

report_lines.append("")


for shift_type in SHIFT_TYPES:

    report_lines.append(
        "-" * 70
    )

    report_lines.append(
        shift_type.upper()
    )

    report_lines.append(
        "-" * 70
    )

    g = trajectory.loc[
        trajectory[
            "shift_type"
        ] == shift_type
    ]

    report_lines.append(
        f"D x A x M trajectory cells: {len(g)}"
    )

    report_lines.append(
        (
            "Fraction of cells with negative mean "
            f"delta-AUC: "
            f"{(g['mean_delta_auc_mse'] < 0).mean():.3f}"
        )
    )

    report_lines.append(
        (
            "Fraction with >=80% of seeds improved: "
            f"{(g['fraction_improved'] >= 0.8).mean():.3f}"
        )
    )

    report_lines.append(
        (
            "Fraction with all 10 seeds improved: "
            f"{g['all_seeds_improved'].mean():.3f}"
        )
    )

    report_lines.append(
        (
            "Cross-cell SD of mean delta-AUC: "
            f"{g['mean_delta_auc_mse'].std(ddof=1):.6f}"
        )
    )

    most_negative = g.loc[
        g[
            "mean_delta_auc_mse"
        ].idxmin()
    ]

    most_positive = g.loc[
        g[
            "mean_delta_auc_mse"
        ].idxmax()
    ]

    report_lines.append("")

    report_lines.append(
        "Largest observed mean trajectory-level reduction:"
    )

    report_lines.append(
        (
            f"  D={most_negative['dataset_id']}, "
            f"A={most_negative['augmentation_id']}, "
            f"M={most_negative['model_config']}, "
            f"mean delta-AUC="
            f"{most_negative['mean_delta_auc_mse']:.6f}, "
            f"fraction seeds improved="
            f"{most_negative['fraction_improved']:.3f}"
        )
    )

    report_lines.append("")

    report_lines.append(
        "Largest observed mean trajectory-level increase:"
    )

    report_lines.append(
        (
            f"  D={most_positive['dataset_id']}, "
            f"A={most_positive['augmentation_id']}, "
            f"M={most_positive['model_config']}, "
            f"mean delta-AUC="
            f"{most_positive['mean_delta_auc_mse']:.6f}, "
            f"fraction seeds improved="
            f"{most_positive['fraction_improved']:.3f}"
        )
    )

    report_lines.append("")

    c = concentration.loc[
        concentration[
            "shift_type"
        ] == shift_type
    ].sort_values(
        "shift_value"
    )

    report_lines.append(
        "Cross-cell dispersion by severity:"
    )

    for _, row in c.iterrows():

        report_lines.append(
            (
                f"  shift={row['shift_value']:.1f}: "
                f"SD={row['sd_across_cells']:.6f}, "
                f"range={row['range_cell_effect']:.6f}, "
                f"fraction mean-improved="
                f"{row['fraction_cells_improved']:.3f}"
            )
        )

    report_lines.append("")


report_lines.append(
    "=" * 70
)

report_lines.append(
    "INTERPRETATION BOUNDARY"
)

report_lines.append(
    "=" * 70
)

report_lines.append("")

report_lines.append(
    "Observed extreme cells are descriptive properties of "
    "this frozen synthetic CEA experiment. They must not be "
    "interpreted as universal best or worst D/A/M choices."
)

report_lines.append("")

report_lines.append(
    "The confirmatory status of Phase 5 remains determined "
    "by the previously frozen H5.5 and H5.6 analyses."
)

report_lines.append("")

report_lines.append(
    "The present analysis is intended to identify whether "
    "the confirmed interaction patterns are broad, localized, "
    "severity-dependent, and directionally stable across seeds."
)


REPORT_FILE = (
    OUTPUT_DIR
    / "phase5_effect_pattern_report.txt"
)


REPORT_FILE.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8",
)


# ============================================================
# Final
# ============================================================

print_section(
    "COMPLETE"
)


print(
    "Generated:"
)

for path in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  {path.name}"
    )


print()

print(
    "PHASE 5 EFFECT-PATTERN ANALYSIS COMPLETE"
)

print()

print(
    "No model training was performed."
)

print(
    "No confirmatory hypothesis was changed."
)

print(
    "H5.5 and H5.6 remain frozen."
)