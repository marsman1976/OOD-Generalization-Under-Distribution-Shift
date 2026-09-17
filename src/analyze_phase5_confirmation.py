from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm


# ============================================================
# PHASE 5 CONFIRMATION ANALYSIS
#
# Primary:
# H5.5 — D × A × M interaction in trajectory-level ΔAUC
#
# Key secondary:
# H5.6 — D × A × M × V severity-dependent interaction
#
# IMPORTANT:
# - Confirmation results are never modified.
# - Temperature and humidity are analyzed separately.
# - A0 is the paired augmentation reference.
# - Δ < 0 means augmentation reduced error.
# - Seed is the replication/blocking unit.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "results"
    / "experiment_5"
    / "confirmation"
)

INPUT_FILE = (
    INPUT_DIR
    / "master_results.csv"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Frozen design
# ============================================================

DATASETS = [
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
]

AUGMENTATIONS = [
    "A0", "A1", "A2", "A3", "A4",
]

NON_REFERENCE_AUGMENTATIONS = [
    "A1", "A2", "A3", "A4",
]

MODELS = [
    "M0", "M1", "M2",
]

SEEDS = [
    121, 132, 143, 154, 165,
    176, 187, 198, 209, 220,
]

SHIFT_FAMILIES = [
    "temperature",
    "humidity",
]


# ============================================================
# Utility
# ============================================================

def save_csv(df, filename):

    path = OUTPUT_DIR / filename

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


def safe_float(x):

    try:
        return float(x)

    except Exception:
        return np.nan


# ============================================================
# Load confirmation data
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print("=" * 76)
    print("PHASE 5 CONFIRMATION ANALYSIS")
    print("=" * 76)

    print(
        "Input records:",
        len(df),
    )

    if len(df) != 30000:

        raise RuntimeError(
            "Expected exactly 30000 confirmation records."
        )

    return df


# ============================================================
# Extract EXP1C severity trajectories
# ============================================================

def prepare_exp1c(df):

    exp1c = df.loc[
        df["experiment"] == "EXP1C"
    ].copy()

    exp1c = exp1c.loc[
        exp1c["shift_type"].isin(
            SHIFT_FAMILIES
        )
    ].copy()

    exp1c["severity"] = (
        exp1c["severity"]
        .astype(int)
    )

    exp1c["shift_value"] = (
        exp1c["shift_value"]
        .astype(float)
    )

    expected = 19500

    if len(exp1c) != expected:

        raise RuntimeError(
            f"Expected {expected} EXP1C rows; "
            f"found {len(exp1c)}."
        )

    return exp1c


# ============================================================
# AUC
#
# Trapezoidal integration over PHYSICAL shift values:
#
# Temperature:
# 23,25,27,29,31,33
#
# Humidity:
# 65,70,75,80,85,90,92
#
# One AUC per:
#
# D × A × M × seed × shift family
# ============================================================

def compute_auc_table(exp1c):

    records = []

    grouping = [
        "dataset_id",
        "augmentation_id",
        "model_config",
        "seed",
        "shift_type",
    ]

    for keys, g in exp1c.groupby(
        grouping,
        sort=True,
    ):

        (
            dataset_id,
            augmentation_id,
            model_config,
            seed,
            shift_type,
        ) = keys

        g = g.sort_values(
            "shift_value"
        )

        x = (
            g["shift_value"]
            .to_numpy(
                dtype=float
            )
        )

        y_mse = (
            g["mse"]
            .to_numpy(
                dtype=float
            )
        )

        y_mae = (
            g["mae"]
            .to_numpy(
                dtype=float
            )
        )

        expected_points = (
            6
            if shift_type == "temperature"
            else 7
        )

        if len(g) != expected_points:

            raise RuntimeError(
                "Unexpected trajectory length for "
                f"{dataset_id}, {augmentation_id}, "
                f"{model_config}, seed={seed}, "
                f"{shift_type}: {len(g)}"
            )

        # NumPy compatibility:
        # np.trapezoid is preferred in newer NumPy,
        # np.trapz works in older versions.

        if hasattr(
            np,
            "trapezoid",
        ):

            auc_mse = np.trapezoid(
                y_mse,
                x,
            )

            auc_mae = np.trapezoid(
                y_mae,
                x,
            )

        else:

            auc_mse = np.trapz(
                y_mse,
                x,
            )

            auc_mae = np.trapz(
                y_mae,
                x,
            )

        records.append({

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
                len(g),

            "shift_min":
                float(x.min()),

            "shift_max":
                float(x.max()),

            "auc_mse":
                float(auc_mse),

            "auc_mae":
                float(auc_mae),
        })

    auc = pd.DataFrame(
        records
    )

    expected_rows = (
        10
        * 5
        * 3
        * 10
        * 2
    )

    if len(auc) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} AUC rows; "
            f"found {len(auc)}."
        )

    return auc


# ============================================================
# Construct paired ΔAUC
#
# ΔAUC =
#
# AUC(D,A,M,seed)
# -
# AUC(D,A0,M,seed)
#
# Negative:
# augmentation reduced accumulated error.
#
# IMPORTANT:
# A0 itself is NOT included as an ordinary zero-valued
# augmentation level in the factorial analysis.
# ============================================================

def compute_delta_auc(auc):

    reference = (
        auc.loc[
            auc["augmentation_id"] == "A0",
            [
                "dataset_id",
                "model_config",
                "seed",
                "shift_type",
                "auc_mse",
                "auc_mae",
            ],
        ]
        .copy()
    )

    reference = reference.rename(
        columns={
            "auc_mse":
                "auc_mse_A0",

            "auc_mae":
                "auc_mae_A0",
        }
    )

    treated = (
        auc.loc[
            auc["augmentation_id"].isin(
                NON_REFERENCE_AUGMENTATIONS
            )
        ]
        .copy()
    )

    paired = treated.merge(

        reference,

        on=[
            "dataset_id",
            "model_config",
            "seed",
            "shift_type",
        ],

        how="left",

        validate="many_to_one",
    )

    if (
        paired["auc_mse_A0"]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing A0 AUC reference."
        )

    paired["delta_auc_mse"] = (
        paired["auc_mse"]
        - paired["auc_mse_A0"]
    )

    paired["delta_auc_mae"] = (
        paired["auc_mae"]
        - paired["auc_mae_A0"]
    )

    expected_rows = (
        10
        * 4
        * 3
        * 10
        * 2
    )

    if len(paired) != expected_rows:

        raise RuntimeError(
            f"Expected {expected_rows} paired AUC rows; "
            f"found {len(paired)}."
        )

    return paired


# ============================================================
# Descriptive ΔAUC summary
# ============================================================

def summarize_delta_auc(paired):

    summary = (

        paired

        .groupby([
            "shift_type",
            "augmentation_id",
            "model_config",
        ])[
            "delta_auc_mse"
        ]

        .agg(
            n="size",
            mean="mean",
            median="median",
            sd="std",
            minimum="min",
            maximum="max",
        )

        .reset_index()
    )

    better = (

        paired

        .assign(
            improved=(
                paired[
                    "delta_auc_mse"
                ] < 0
            )
        )

        .groupby([
            "shift_type",
            "augmentation_id",
            "model_config",
        ])[
            "improved"
        ]

        .mean()

        .reset_index(
            name="fraction_improved"
        )
    )

    summary = summary.merge(

        better,

        on=[
            "shift_type",
            "augmentation_id",
            "model_config",
        ],

        how="left",
    )

    return summary


# ============================================================
# H5.5 PRIMARY
#
# Trajectory-level:
#
# ΔAUC ~ D * A * M + seed
#
# Separate:
# - temperature
# - humidity
#
# A = A1-A4 only.
#
# We use a nested partial F test for the fixed factorial
# D × A × M term, with seed included as a blocking factor.
#
# This is the trajectory-level omnibus effect-description
# model.
# ============================================================

def h55_fixed_block_test(
    paired,
    shift_type,
):

    d = paired.loc[
        paired["shift_type"] == shift_type
    ].copy()

    full_formula = (
        "delta_auc_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) * "
        "C(model_config)"
    )

    reduced_formula = (
        "delta_auc_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) + "
        "C(dataset_id) * "
        "C(model_config) + "
        "C(augmentation_id) * "
        "C(model_config)"
    )

    full = smf.ols(
        full_formula,
        data=d,
    ).fit()

    reduced = smf.ols(
        reduced_formula,
        data=d,
    ).fit()

    f_stat, p_value, df_diff = (
        full.compare_f_test(
            reduced
        )
    )

    result = {

        "shift_type":
            shift_type,

        "hypothesis":
            "H5.5",

        "effect":
            "D x A x M",

        "n":
            len(d),

        "n_seeds":
            d["seed"].nunique(),

        "f_statistic":
            float(f_stat),

        "p_value":
            float(p_value),

        "df_difference":
            int(df_diff),

        "full_df_resid":
            float(full.df_resid),

        "analysis":
            "fixed-block nested partial F",

        "primary_metric":
            "delta_auc_mse",
    }

    return (
        result,
        full,
        reduced,
    )


# ============================================================
# H5.5 SEED-RESPECTING PERMUTATION SENSITIVITY
#
# This is deliberately performed at the trajectory level.
#
# Within each seed × D × M block, augmentation labels
# A1-A4 are permuted together.
#
# This preserves:
# - seed structure
# - D structure
# - M structure
# - number of observations
#
# and breaks the association between augmentation identity
# and ΔAUC.
#
# Statistic:
# partial-F statistic for D × A × M.
#
# This is a computational sensitivity analysis for H5.5.
# ============================================================

def h55_permutation_test(
    paired,
    shift_type,
    n_permutations=2000,
    random_seed=20260916,
):

    d = paired.loc[
        paired["shift_type"] == shift_type
    ].copy()

    full_formula = (
        "delta_auc_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) * "
        "C(model_config)"
    )

    reduced_formula = (
        "delta_auc_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) + "
        "C(dataset_id) * "
        "C(model_config) + "
        "C(augmentation_id) * "
        "C(model_config)"
    )

    observed_full = smf.ols(
        full_formula,
        data=d,
    ).fit()

    observed_reduced = smf.ols(
        reduced_formula,
        data=d,
    ).fit()

    observed_f = (
        observed_full.compare_f_test(
            observed_reduced
        )[0]
    )

    rng = np.random.default_rng(
        random_seed
    )

    permutation_statistics = []

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Augmentation labels are permuted at the D × seed level,
    # and the SAME A-label mapping is applied across M0/M1/M2.
    #
    # This preserves the fact that the same augmented dataset
    # was shared across the three model configurations.
    # --------------------------------------------------------

    blocks = list(
        d.groupby([
            "dataset_id",
            "seed",
        ]).groups.items()
    )

    augmentation_array = np.array(
        NON_REFERENCE_AUGMENTATIONS
    )

    for permutation_id in range(
        n_permutations
    ):

        permuted = d.copy()

        for _, indices in blocks:

            mapping_values = (
                rng.permutation(
                    augmentation_array
                )
            )

            mapping = dict(
                zip(
                    augmentation_array,
                    mapping_values,
                )
            )

            original_labels = (
                permuted.loc[
                    indices,
                    "augmentation_id",
                ]
                .copy()
            )

            permuted.loc[
                indices,
                "augmentation_id",
            ] = (
                original_labels
                .map(mapping)
                .to_numpy()
            )

        full_p = smf.ols(
            full_formula,
            data=permuted,
        ).fit()

        reduced_p = smf.ols(
            reduced_formula,
            data=permuted,
        ).fit()

        f_p = (
            full_p.compare_f_test(
                reduced_p
            )[0]
        )

        permutation_statistics.append(
            float(f_p)
        )

    permutation_statistics = np.array(
        permutation_statistics
    )

    p_value = (
        1
        + np.sum(
            permutation_statistics
            >= observed_f
        )
    ) / (
        n_permutations + 1
    )

    summary = {

        "shift_type":
            shift_type,

        "hypothesis":
            "H5.5",

        "effect":
            "D x A x M",

        "observed_f":
            float(observed_f),

        "n_permutations":
            int(n_permutations),

        "permutation_p":
            float(p_value),

        "random_seed":
            int(random_seed),

        "permutation_unit":
            (
                "augmentation labels permuted within "
                "dataset x seed; same mapping across models"
            ),
    }

    distribution = pd.DataFrame({

        "permutation_id":
            np.arange(
                n_permutations
            ),

        "f_statistic":
            permutation_statistics,
    })

    return (
        summary,
        distribution,
    )


# ============================================================
# Construct pointwise ΔMSE
#
# For H5.6:
#
# ΔMSE(D,A,M,V)
# =
# MSE(D,A,M,V) - MSE(D,A0,M,V)
# ============================================================

def compute_pointwise_delta(
    exp1c,
):

    reference = (

        exp1c.loc[
            exp1c["augmentation_id"] == "A0",
            [
                "dataset_id",
                "model_config",
                "seed",
                "shift_type",
                "severity",
                "shift_value",
                "mse",
                "mae",
            ],
        ]

        .copy()
    )

    reference = reference.rename(
        columns={
            "mse":
                "mse_A0",

            "mae":
                "mae_A0",
        }
    )

    treated = (

        exp1c.loc[
            exp1c[
                "augmentation_id"
            ].isin(
                NON_REFERENCE_AUGMENTATIONS
            )
        ]

        .copy()
    )

    paired = treated.merge(

        reference,

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

    paired["delta_mse"] = (
        paired["mse"]
        - paired["mse_A0"]
    )

    paired["delta_mae"] = (
        paired["mae"]
        - paired["mae_A0"]
    )

    return paired


# ============================================================
# Standardized severity
#
# Standardize separately inside each shift family.
# ============================================================

def add_severity_z(pointwise):

    frames = []

    for shift_type in SHIFT_FAMILIES:

        d = pointwise.loc[
            pointwise["shift_type"]
            == shift_type
        ].copy()

        values = (
            d["shift_value"]
            .astype(float)
        )

        mean = values.mean()
        sd = values.std(
            ddof=0
        )

        d["severity_z"] = (
            (values - mean)
            / sd
        )

        frames.append(
            d
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# H5.6
#
# Key secondary severity interaction:
#
# ΔMSE ~
# D * A * M * severity_z
# + seed
#
# We test whether adding the D×A×M×V term improves the model.
#
# Separate temperature/humidity.
# ============================================================

def h56_test(
    pointwise,
    shift_type,
):

    d = pointwise.loc[
        pointwise["shift_type"]
        == shift_type
    ].copy()

    full_formula = (
        "delta_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) * "
        "C(model_config) * "
        "severity_z"
    )

    reduced_formula = (
        "delta_mse ~ "
        "C(seed) + "
        "C(dataset_id) * "
        "C(augmentation_id) * "
        "C(model_config) + "
        "C(dataset_id) * "
        "C(augmentation_id) * "
        "severity_z + "
        "C(dataset_id) * "
        "C(model_config) * "
        "severity_z + "
        "C(augmentation_id) * "
        "C(model_config) * "
        "severity_z"
    )

    full = smf.ols(
        full_formula,
        data=d,
    ).fit()

    reduced = smf.ols(
        reduced_formula,
        data=d,
    ).fit()

    f_stat, p_value, df_diff = (
        full.compare_f_test(
            reduced
        )
    )

    return {

        "shift_type":
            shift_type,

        "hypothesis":
            "H5.6",

        "effect":
            "D x A x M x V",

        "n":
            len(d),

        "n_seeds":
            d["seed"].nunique(),

        "f_statistic":
            float(f_stat),

        "p_value":
            float(p_value),

        "df_difference":
            int(df_diff),

        "full_df_resid":
            float(full.df_resid),

        "analysis":
            "fixed-block nested partial F",

        "metric":
            "delta_mse",
    }


# ============================================================
# Per-severity D × A × M sensitivity
#
# Secondary only.
#
# At each severity point:
#
# ΔMSE ~ D * A * M + seed
#
# Compare against model without D×A×M.
#
# Multiplicity:
# FDR + Bonferroni within shift family.
# ============================================================

def per_severity_tests(
    pointwise,
):

    records = []

    for shift_type in SHIFT_FAMILIES:

        family = pointwise.loc[
            pointwise["shift_type"]
            == shift_type
        ].copy()

        severity_values = (
            family[
                [
                    "severity",
                    "shift_value",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "severity"
            )
        )

        for _, row in severity_values.iterrows():

            severity = int(
                row["severity"]
            )

            shift_value = float(
                row["shift_value"]
            )

            d = family.loc[
                family["severity"]
                == severity
            ].copy()

            full_formula = (
                "delta_mse ~ "
                "C(seed) + "
                "C(dataset_id) * "
                "C(augmentation_id) * "
                "C(model_config)"
            )

            reduced_formula = (
                "delta_mse ~ "
                "C(seed) + "
                "C(dataset_id) * "
                "C(augmentation_id) + "
                "C(dataset_id) * "
                "C(model_config) + "
                "C(augmentation_id) * "
                "C(model_config)"
            )

            full = smf.ols(
                full_formula,
                data=d,
            ).fit()

            reduced = smf.ols(
                reduced_formula,
                data=d,
            ).fit()

            f_stat, p_value, df_diff = (
                full.compare_f_test(
                    reduced
                )
            )

            records.append({

                "shift_type":
                    shift_type,

                "severity":
                    severity,

                "shift_value":
                    shift_value,

                "n":
                    len(d),

                "f_statistic":
                    float(f_stat),

                "p_value":
                    float(p_value),

                "df_difference":
                    int(df_diff),
            })

    results = pd.DataFrame(
        records
    )

    results["p_fdr"] = np.nan
    results["p_bonferroni"] = np.nan

    for shift_type in SHIFT_FAMILIES:

        mask = (
            results["shift_type"]
            == shift_type
        )

        p = (
            results.loc[
                mask,
                "p_value",
            ]
            .to_numpy()
        )

        results.loc[
            mask,
            "p_fdr",
        ] = multipletests(
            p,
            method="fdr_bh",
        )[1]

        results.loc[
            mask,
            "p_bonferroni",
        ] = multipletests(
            p,
            method="bonferroni",
        )[1]

    return results


# ============================================================
# Cell-level trajectory summaries
# ============================================================

def cell_auc_summary(
    paired,
):

    return (

        paired

        .groupby([
            "shift_type",
            "dataset_id",
            "augmentation_id",
            "model_config",
        ])[
            "delta_auc_mse"
        ]

        .agg(
            n_seeds="size",
            mean_delta_auc="mean",
            median_delta_auc="median",
            sd_delta_auc="std",
        )

        .reset_index()
    )


# ============================================================
# Severity descriptive summary
# ============================================================

def severity_summary(
    pointwise,
):

    summary = (

        pointwise

        .groupby([
            "shift_type",
            "severity",
            "shift_value",
            "augmentation_id",
            "model_config",
        ])[
            "delta_mse"
        ]

        .agg(
            n="size",
            mean_delta_mse="mean",
            median_delta_mse="median",
            sd_delta_mse="std",
        )

        .reset_index()
    )

    return summary


# ============================================================
# Main
# ============================================================

def main():

    df = load_data()

    exp1c = prepare_exp1c(
        df
    )

    print()
    print(
        "EXP1C records:",
        len(exp1c),
    )


    # ========================================================
    # AUC
    # ========================================================

    auc = compute_auc_table(
        exp1c
    )

    save_csv(
        auc,
        "phase5_auc.csv",
    )


    # ========================================================
    # Paired ΔAUC
    # ========================================================

    paired_auc = compute_delta_auc(
        auc
    )

    save_csv(
        paired_auc,
        "phase5_delta_auc.csv",
    )


    descriptive_auc = (
        summarize_delta_auc(
            paired_auc
        )
    )

    save_csv(
        descriptive_auc,
        "phase5_delta_auc_summary.csv",
    )


    cell_summary = (
        cell_auc_summary(
            paired_auc
        )
    )

    save_csv(
        cell_summary,
        "phase5_delta_auc_cells.csv",
    )


    # ========================================================
    # H5.5 primary fixed-block omnibus
    # ========================================================

    h55_records = []

    h55_models = {}

    for shift_type in SHIFT_FAMILIES:

        (
            result,
            full,
            reduced,
        ) = h55_fixed_block_test(

            paired_auc,
            shift_type,
        )

        h55_records.append(
            result
        )

        h55_models[
            shift_type
        ] = full

    h55 = pd.DataFrame(
        h55_records
    )


    # multiplicity across the two shift families

    h55["p_fdr"] = multipletests(
        h55["p_value"],
        method="fdr_bh",
    )[1]

    h55["p_bonferroni"] = (
        multipletests(
            h55["p_value"],
            method="bonferroni",
        )[1]
    )

    save_csv(
        h55,
        "phase5_h55_primary.csv",
    )


    # ========================================================
    # H5.5 permutation sensitivity
    # ========================================================

    permutation_summaries = []

    for i, shift_type in enumerate(
        SHIFT_FAMILIES
    ):

        print()
        print(
            "Running H5.5 permutation sensitivity:",
            shift_type,
        )

        summary, distribution = (
            h55_permutation_test(

                paired_auc,

                shift_type,

                n_permutations=2000,

                random_seed=(
                    20260916 + i
                ),
            )
        )

        permutation_summaries.append(
            summary
        )

        save_csv(

            distribution,

            (
                "phase5_h55_permutation_"
                f"{shift_type}.csv"
            ),
        )

    permutation_summary = (
        pd.DataFrame(
            permutation_summaries
        )
    )

    permutation_summary["p_fdr"] = (
        multipletests(
            permutation_summary[
                "permutation_p"
            ],
            method="fdr_bh",
        )[1]
    )

    permutation_summary[
        "p_bonferroni"
    ] = multipletests(
        permutation_summary[
            "permutation_p"
        ],
        method="bonferroni",
    )[1]

    save_csv(
        permutation_summary,
        "phase5_h55_permutation_summary.csv",
    )


    # ========================================================
    # Pointwise ΔMSE
    # ========================================================

    pointwise = (
        compute_pointwise_delta(
            exp1c
        )
    )

    pointwise = add_severity_z(
        pointwise
    )

    save_csv(
        pointwise,
        "phase5_pointwise_delta.csv",
    )


    # ========================================================
    # H5.6
    # ========================================================

    h56_records = []

    for shift_type in SHIFT_FAMILIES:

        result = h56_test(
            pointwise,
            shift_type,
        )

        h56_records.append(
            result
        )

    h56 = pd.DataFrame(
        h56_records
    )

    h56["p_fdr"] = multipletests(
        h56["p_value"],
        method="fdr_bh",
    )[1]

    h56["p_bonferroni"] = (
        multipletests(
            h56["p_value"],
            method="bonferroni",
        )[1]
    )

    save_csv(
        h56,
        "phase5_h56_secondary.csv",
    )


    # ========================================================
    # Per-severity sensitivity
    # ========================================================

    per_severity = (
        per_severity_tests(
            pointwise
        )
    )

    save_csv(
        per_severity,
        "phase5_per_severity_DxAxM.csv",
    )


    # ========================================================
    # Severity descriptive summaries
    # ========================================================

    sev_summary = (
        severity_summary(
            pointwise
        )
    )

    save_csv(
        sev_summary,
        "phase5_severity_summary.csv",
    )


    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 76)
    print("H5.5 PRIMARY — TRAJECTORY D × A × M")
    print("=" * 76)

    print(
        h55[
            [
                "shift_type",
                "f_statistic",
                "p_value",
                "p_fdr",
                "p_bonferroni",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 76)
    print("H5.5 SEED-RESPECTING PERMUTATION SENSITIVITY")
    print("=" * 76)

    print(
        permutation_summary[
            [
                "shift_type",
                "observed_f",
                "permutation_p",
                "p_fdr",
                "p_bonferroni",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 76)
    print("H5.6 — D × A × M × V")
    print("=" * 76)

    print(
        h56[
            [
                "shift_type",
                "f_statistic",
                "p_value",
                "p_fdr",
                "p_bonferroni",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print("=" * 76)
    print("PHASE 5 CONFIRMATION ANALYSIS COMPLETE")
    print("=" * 76)

    print()
    print(
        "Analysis directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()