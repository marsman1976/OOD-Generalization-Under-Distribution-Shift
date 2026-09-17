from pathlib import Path
import json
import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# PHASE 4 — FORMAL INTERACTION ANALYSIS
# ============================================================
#
# Primary hypothesis:
#
# H4.5:
#     Augmentation × Model × Shift Severity
#
# Formal idea:
#
#     delta_mse ~ A * M * V
#
# where:
#
#     delta_mse = MSE(A) - MSE(A0)
#
# Negative delta:
#     augmentation improves performance.
#
# Positive delta:
#     augmentation harms performance.
#
#
# IMPORTANT DESIGN FEATURE
# ------------------------
#
# D is fixed to D00.
#
# There are 10 independent training seeds.
#
# Therefore seed — not every severity observation —
# is treated as the primary replication unit.
#
#
# PRIMARY H4.5 INFERENCE
# ----------------------
#
# For each seed:
#
# 1. Estimate the delta-MSE severity slope separately
#    for every A × M combination.
#
# 2. Construct six A × M difference-in-differences
#    of severity slopes:
#
#    A2 vs A1 crossed with M1 vs M0
#    A2 vs A1 crossed with M2 vs M0
#    A3 vs A1 crossed with M1 vs M0
#    A3 vs A1 crossed with M2 vs M0
#    A4 vs A1 crossed with M1 vs M0
#    A4 vs A1 crossed with M2 vs M0
#
# 3. This produces one six-dimensional interaction
#    contrast vector per seed.
#
# 4. Perform an exact sign-flip test across the
#    10 seed-level vectors.
#
# Number of exact sign patterns:
#
#     2^10 = 1024
#
#
# This is the PRIMARY seed-respecting H4.5 analysis.
#
#
# ADDITIONAL ANALYSES
# -------------------
#
# - OLS coefficient table
# - cluster-robust-by-seed sensitivity analysis
# - H4.1 descriptive seed-level augmentation effects
# - H4.2/H4.3/H4.4 secondary nested-model summaries
# - per-severity A × M sensitivity analyses
#
#
# Temperature and humidity are analyzed separately.
# ============================================================


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_DIR = (
    ROOT
    / "results"
    / "experiment_4"
    / "confirmation"
    / "analysis"
)

PAIRED_FILE = (
    ANALYSIS_DIR
    / "paired_augmentation_effects.csv"
)

OUTPUT_DIR = (
    ANALYSIS_DIR
    / "formal_interactions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

PRIMARY_FILE = (
    OUTPUT_DIR
    / "primary_H45_results.csv"
)

PRIMARY_CONTRAST_FILE = (
    OUTPUT_DIR
    / "H45_seed_level_contrasts.csv"
)

PRIMARY_COMPONENT_FILE = (
    OUTPUT_DIR
    / "H45_component_tests.csv"
)

SENSITIVITY_FILE = (
    OUTPUT_DIR
    / "cluster_robust_sensitivity.csv"
)

COEFFICIENT_FILE = (
    OUTPUT_DIR
    / "regression_coefficients.csv"
)

H41_FILE = (
    OUTPUT_DIR
    / "H41_augmentation_effects.csv"
)

H42_FILE = (
    OUTPUT_DIR
    / "H42_AxM_results.csv"
)

H43_FILE = (
    OUTPUT_DIR
    / "H43_AxV_results.csv"
)

H44_FILE = (
    OUTPUT_DIR
    / "H44_MxV_results.csv"
)

PER_SEVERITY_FILE = (
    OUTPUT_DIR
    / "per_severity_AxM_tests.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "phase4_formal_report.json"
)


# ============================================================
# Frozen configuration
# ============================================================

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

SHIFT_TYPES = [
    "temperature",
    "humidity",
]

ALPHA = 0.05


# ============================================================
# Load and verify data
# ============================================================

def load_data():

    if not PAIRED_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{PAIRED_FILE}"
        )

    df = pd.read_csv(
        PAIRED_FILE
    )

    required_columns = {
        "dataset_id",
        "augmentation_id",
        "model_config",
        "seed",
        "experiment",
        "environment",
        "shift_type",
        "severity",
        "shift_value",
        "delta_mse",
        "delta_mae",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing required columns: "
            + str(sorted(missing))
        )

    # Phase 4 formal interaction analysis
    # uses EXP1C severity trajectories.

    df = df[
        df["experiment"]
        == "EXP1C"
    ].copy()

    assert len(df) == 1560, (
        f"Expected 1560 EXP1C paired rows, "
        f"found {len(df)}"
    )

    assert set(
        df["dataset_id"].unique()
    ) == {"D00"}

    assert set(
        df["augmentation_id"].unique()
    ) == set(AUGMENTATIONS)

    assert set(
        df["model_config"].unique()
    ) == set(MODELS)

    assert set(
        df["seed"].unique()
    ) == set(SEEDS)

    assert set(
        df["shift_type"].unique()
    ) == set(SHIFT_TYPES)

    assert (
        df["delta_mse"]
        .notna()
        .all()
    )

    assert np.isfinite(
        df["delta_mse"].to_numpy()
    ).all()

    temp_n = len(
        df[
            df["shift_type"]
            == "temperature"
        ]
    )

    humid_n = len(
        df[
            df["shift_type"]
            == "humidity"
        ]
    )

    assert temp_n == 720, (
        f"Expected 720 temperature pairs, "
        f"found {temp_n}"
    )

    assert humid_n == 840, (
        f"Expected 840 humidity pairs, "
        f"found {humid_n}"
    )

    return df


# ============================================================
# Standardize severity within shift family
# ============================================================

def add_standardized_severity(df):

    frames = []

    for shift_type in SHIFT_TYPES:

        part = df[
            df["shift_type"]
            == shift_type
        ].copy()

        severity = (
            part["severity"]
            .astype(float)
        )

        mean_v = severity.mean()

        sd_v = severity.std(
            ddof=0
        )

        if sd_v <= 0:

            raise RuntimeError(
                f"Invalid severity SD for "
                f"{shift_type}"
            )

        part["severity_z"] = (
            severity - mean_v
        ) / sd_v

        frames.append(
            part
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    assert (
        result["severity_z"]
        .notna()
        .all()
    )

    return result


# ============================================================
# Linear slope helper
# ============================================================

def simple_slope(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:

        raise RuntimeError(
            "Not enough observations "
            "to estimate slope."
        )

    x_centered = (
        x - np.mean(x)
    )

    denominator = np.sum(
        x_centered ** 2
    )

    if denominator <= 0:

        raise RuntimeError(
            "Cannot estimate slope: "
            "zero predictor variance."
        )

    numerator = np.sum(
        x_centered
        * (
            y - np.mean(y)
        )
    )

    return float(
        numerator / denominator
    )


# ============================================================
# H4.5:
# Build seed-level A × M × V contrast vectors
# ============================================================

def build_H45_seed_contrasts(
    data,
    shift_type,
):

    shift_df = data[
        data["shift_type"]
        == shift_type
    ].copy()

    expected_levels = (
        6
        if shift_type == "temperature"
        else 7
    )

    records = []

    contrast_names = []

    # Reference augmentation:
    #
    # A1
    #
    # Reference model:
    #
    # M0
    #
    # We therefore construct:
    #
    # (slope(A,M) - slope(A,M0))
    # -
    # (slope(A1,M) - slope(A1,M0))
    #
    # for A2/A3/A4 and M1/M2.

    for augmentation_id in [
        "A2",
        "A3",
        "A4",
    ]:

        for model_config in [
            "M1",
            "M2",
        ]:

            contrast_names.append(
                f"{augmentation_id}_vs_A1"
                f"__{model_config}_vs_M0"
            )

    for seed in SEEDS:

        seed_df = shift_df[
            shift_df["seed"]
            == seed
        ].copy()

        slopes = {}

        for augmentation_id in AUGMENTATIONS:

            for model_config in MODELS:

                cell = seed_df[
                    (
                        seed_df[
                            "augmentation_id"
                        ]
                        == augmentation_id
                    )
                    &
                    (
                        seed_df[
                            "model_config"
                        ]
                        == model_config
                    )
                ].sort_values(
                    "severity"
                )

                if len(cell) != expected_levels:

                    raise RuntimeError(
                        f"Unexpected number of "
                        f"severity observations for "
                        f"{shift_type}, seed={seed}, "
                        f"A={augmentation_id}, "
                        f"M={model_config}: "
                        f"{len(cell)}"
                    )

                slope = simple_slope(
                    cell["severity_z"],
                    cell["delta_mse"],
                )

                slopes[
                    (
                        augmentation_id,
                        model_config,
                    )
                ] = slope

        record = {
            "shift_type":
                shift_type,

            "seed":
                seed,
        }

        for augmentation_id in [
            "A2",
            "A3",
            "A4",
        ]:

            for model_config in [
                "M1",
                "M2",
            ]:

                contrast_name = (
                    f"{augmentation_id}_vs_A1"
                    f"__{model_config}_vs_M0"
                )

                contrast = (

                    slopes[
                        (
                            augmentation_id,
                            model_config,
                        )
                    ]

                    - slopes[
                        (
                            augmentation_id,
                            "M0",
                        )
                    ]

                    - slopes[
                        (
                            "A1",
                            model_config,
                        )
                    ]

                    + slopes[
                        (
                            "A1",
                            "M0",
                        )
                    ]

                )

                record[
                    contrast_name
                ] = float(
                    contrast
                )

        records.append(
            record
        )

    result = pd.DataFrame(
        records
    )

    assert len(result) == 10

    return (
        result,
        contrast_names,
    )


# ============================================================
# Hotelling-style multivariate statistic
# ============================================================

def multivariate_statistic(
    matrix,
):

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    n, p = matrix.shape

    mean_vector = np.mean(
        matrix,
        axis=0,
    )

    covariance = np.cov(
        matrix,
        rowvar=False,
        ddof=1,
    )

    covariance = np.atleast_2d(
        covariance
    )

    covariance_inverse = np.linalg.pinv(
        covariance
    )

    statistic = (

        n
        * mean_vector.T
        @ covariance_inverse
        @ mean_vector

    )

    condition_number = np.linalg.cond(
        covariance
    )

    return (
        float(statistic),
        mean_vector,
        covariance,
        float(condition_number),
    )


# ============================================================
# Generate all exact sign patterns
#
# 10 seeds:
#
# 2^10 = 1024
# ============================================================

def all_sign_patterns(
    n_seeds,
):

    return np.asarray(
        list(
            itertools.product(
                [-1.0, 1.0],
                repeat=n_seeds,
            )
        ),
        dtype=float,
    )


# ============================================================
# Exact multivariate sign-flip test
# ============================================================

def exact_multivariate_signflip(
    contrast_matrix,
):

    matrix = np.asarray(
        contrast_matrix,
        dtype=float,
    )

    n_seeds = matrix.shape[0]

    observed_statistic, mean_vector, covariance, condition_number = (
        multivariate_statistic(
            matrix
        )
    )

    sign_patterns = (
        all_sign_patterns(
            n_seeds
        )
    )

    permutation_statistics = []

    for signs in sign_patterns:

        signed_matrix = (
            matrix
            * signs[:, None]
        )

        statistic, _, _, _ = (
            multivariate_statistic(
                signed_matrix
            )
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = np.asarray(
        permutation_statistics,
        dtype=float,
    )

    # Exact permutation p-value.
    #
    # The observed all-positive pattern is already
    # included in the 1024 sign configurations.

    p_value = np.mean(
        permutation_statistics
        >= (
            observed_statistic
            - 1e-12
        )
    )

    return {
        "observed_statistic":
            observed_statistic,

        "exact_p":
            float(p_value),

        "n_sign_patterns":
            int(
                len(sign_patterns)
            ),

        "condition_number":
            condition_number,

        "mean_vector":
            mean_vector,

        "permutation_statistics":
            permutation_statistics,
    }


# ============================================================
# Exact scalar sign-flip test
# ============================================================

def exact_scalar_signflip(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    observed_mean = float(
        np.mean(values)
    )

    observed_abs = abs(
        observed_mean
    )

    signs = all_sign_patterns(
        n
    )

    permuted_means = np.mean(
        signs * values[None, :],
        axis=1,
    )

    p_value = np.mean(
        np.abs(permuted_means)
        >= (
            observed_abs
            - 1e-12
        )
    )

    sd = float(
        np.std(
            values,
            ddof=1,
        )
    )

    se = (
        sd / np.sqrt(n)
        if sd > 0
        else 0.0
    )

    t_like = (
        observed_mean / se
        if se > 0
        else np.nan
    )

    return {
        "n_seeds":
            n,

        "mean_contrast":
            observed_mean,

        "median_contrast":
            float(
                np.median(values)
            ),

        "sd_contrast":
            sd,

        "se_contrast":
            se,

        "t_like":
            float(t_like)
            if np.isfinite(t_like)
            else np.nan,

        "exact_signflip_p":
            float(p_value),

        "fraction_negative":
            float(
                np.mean(
                    values < 0
                )
            ),

        "fraction_positive":
            float(
                np.mean(
                    values > 0
                )
            ),
    }


# ============================================================
# Primary H4.5 analysis
# ============================================================

def analyze_H45(
    data
):

    primary_records = []

    seed_contrast_frames = []

    component_records = []

    for shift_type in SHIFT_TYPES:

        print()
        print(
            "=" * 72
        )
        print(
            f"H4.5 PRIMARY: {shift_type}"
        )
        print(
            "=" * 72
        )

        seed_contrasts, contrast_names = (
            build_H45_seed_contrasts(
                data,
                shift_type,
            )
        )

        seed_contrast_frames.append(
            seed_contrasts
        )

        matrix = (
            seed_contrasts[
                contrast_names
            ]
            .to_numpy(
                dtype=float
            )
        )

        result = (
            exact_multivariate_signflip(
                matrix
            )
        )

        primary_records.append({

            "shift_type":
                shift_type,

            "hypothesis":
                "H4.5",

            "effect":
                "A x M x V",

            "n_seeds":
                10,

            "n_contrasts":
                len(
                    contrast_names
                ),

            "observed_multivariate_statistic":
                result[
                    "observed_statistic"
                ],

            "exact_signflip_p":
                result[
                    "exact_p"
                ],

            "n_exact_sign_patterns":
                result[
                    "n_sign_patterns"
                ],

            "covariance_condition_number":
                result[
                    "condition_number"
                ],

            "significant_raw_0_05":
                bool(
                    result[
                        "exact_p"
                    ]
                    < ALPHA
                ),
        })

        print(
            "Observed multivariate statistic:",
            result[
                "observed_statistic"
            ]
        )

        print(
            "Exact sign-flip p:",
            result[
                "exact_p"
            ]
        )

        print(
            "Exact sign patterns:",
            result[
                "n_sign_patterns"
            ]
        )

        # ----------------------------------------------------
        # Six component interaction contrasts
        # ----------------------------------------------------

        shift_component_records = []

        for contrast_name in contrast_names:

            scalar_result = (
                exact_scalar_signflip(
                    seed_contrasts[
                        contrast_name
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )
            )

            record = {

                "shift_type":
                    shift_type,

                "contrast":
                    contrast_name,

                **scalar_result,
            }

            shift_component_records.append(
                record
            )

        component_df = pd.DataFrame(
            shift_component_records
        )

        # Multiplicity correction across the six
        # component contrasts inside this shift family.

        _, fdr_p, _, _ = (
            multipletests(
                component_df[
                    "exact_signflip_p"
                ],
                alpha=ALPHA,
                method="fdr_bh",
            )
        )

        _, bonf_p, _, _ = (
            multipletests(
                component_df[
                    "exact_signflip_p"
                ],
                alpha=ALPHA,
                method="bonferroni",
            )
        )

        component_df[
            "p_fdr_bh"
        ] = fdr_p

        component_df[
            "p_bonferroni"
        ] = bonf_p

        component_df[
            "significant_fdr"
        ] = (
            component_df[
                "p_fdr_bh"
            ]
            < ALPHA
        )

        component_df[
            "significant_bonferroni"
        ] = (
            component_df[
                "p_bonferroni"
            ]
            < ALPHA
        )

        component_records.extend(
            component_df.to_dict(
                orient="records"
            )
        )

    primary_df = pd.DataFrame(
        primary_records
    )

    # Conservative sensitivity adjustment across
    # the two shift-family primary tests.
    #
    # We retain the raw exact p-values as the main
    # reported shift-family results and additionally
    # report FDR and Bonferroni corrections.

    _, fdr_p, _, _ = (
        multipletests(
            primary_df[
                "exact_signflip_p"
            ],
            alpha=ALPHA,
            method="fdr_bh",
        )
    )

    _, bonf_p, _, _ = (
        multipletests(
            primary_df[
                "exact_signflip_p"
            ],
            alpha=ALPHA,
            method="bonferroni",
        )
    )

    primary_df[
        "p_fdr_bh_across_shift_families"
    ] = fdr_p

    primary_df[
        "p_bonferroni_across_shift_families"
    ] = bonf_p

    primary_df[
        "significant_fdr"
    ] = (
        primary_df[
            "p_fdr_bh_across_shift_families"
        ]
        < ALPHA
    )

    primary_df[
        "significant_bonferroni"
    ] = (
        primary_df[
            "p_bonferroni_across_shift_families"
        ]
        < ALPHA
    )

    seed_contrast_df = pd.concat(
        seed_contrast_frames,
        ignore_index=True,
    )

    component_df = pd.DataFrame(
        component_records
    )

    return (
        primary_df,
        seed_contrast_df,
        component_df,
    )


# ============================================================
# Full regression formula
# ============================================================

FULL_FORMULA = (
    "delta_mse ~ "
    "C(augmentation_id) "
    "* C(model_config) "
    "* severity_z"
)


# ============================================================
# OLS coefficient table
# ============================================================

def regression_coefficients(
    data
):

    records = []

    for shift_type in SHIFT_TYPES:

        part = data[
            data["shift_type"]
            == shift_type
        ].copy()

        model = smf.ols(
            FULL_FORMULA,
            data=part,
        ).fit()

        ci = model.conf_int()

        for parameter in model.params.index:

            records.append({

                "shift_type":
                    shift_type,

                "parameter":
                    parameter,

                "estimate":
                    float(
                        model.params[
                            parameter
                        ]
                    ),

                "std_error_ordinary_ols":
                    float(
                        model.bse[
                            parameter
                        ]
                    ),

                "t_value_ordinary_ols":
                    float(
                        model.tvalues[
                            parameter
                        ]
                    ),

                "p_ordinary_ols":
                    float(
                        model.pvalues[
                            parameter
                        ]
                    ),

                "ci_low_ordinary_ols":
                    float(
                        ci.loc[
                            parameter,
                            0,
                        ]
                    ),

                "ci_high_ordinary_ols":
                    float(
                        ci.loc[
                            parameter,
                            1,
                        ]
                    ),
            })

    return pd.DataFrame(
        records
    )


# ============================================================
# Cluster-robust H4.5 sensitivity
# ============================================================

def cluster_robust_sensitivity(
    data
):

    records = []

    for shift_type in SHIFT_TYPES:

        part = data[
            data["shift_type"]
            == shift_type
        ].copy()

        model = smf.ols(
            FULL_FORMULA,
            data=part,
        ).fit(
            cov_type="cluster",
            cov_kwds={
                "groups":
                    part["seed"]
            },
        )

        names = list(
            model.params.index
        )

        threeway_indices = [
            i
            for i, name
            in enumerate(names)
            if (
                "augmentation_id"
                in name
                and
                "model_config"
                in name
                and
                "severity_z"
                in name
            )
        ]

        if len(
            threeway_indices
        ) == 0:

            raise RuntimeError(
                "Could not identify "
                "three-way coefficients."
            )

        R = np.zeros(
            (
                len(
                    threeway_indices
                ),
                len(names),
            )
        )

        for row, index in enumerate(
            threeway_indices
        ):

            R[
                row,
                index
            ] = 1.0

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            test = model.wald_test(
                R,
                scalar=True,
            )

        statistic = float(
            np.asarray(
                test.statistic
            ).squeeze()
        )

        p_value = float(
            np.asarray(
                test.pvalue
            ).squeeze()
        )

        records.append({

            "shift_type":
                shift_type,

            "effect":
                "A x M x V",

            "n_seed_clusters":
                int(
                    part[
                        "seed"
                    ].nunique()
                ),

            "wald_statistic":
                statistic,

            "cluster_robust_p":
                p_value,

            "significant_0_05":
                bool(
                    p_value < ALPHA
                ),

            "interpretation_role":
                (
                    "Sensitivity only; "
                    "10 clusters."
                ),
        })

    return pd.DataFrame(
        records
    )


# ============================================================
# H4.1
#
# Augmentation effect within fixed model.
#
# Descriptive seed-level summary.
# ============================================================

def analyze_H41(
    data
):

    records = []

    for shift_type in SHIFT_TYPES:

        shift_df = data[
            data["shift_type"]
            == shift_type
        ].copy()

        for augmentation_id in AUGMENTATIONS:

            for model_config in MODELS:

                subset = shift_df[
                    (
                        shift_df[
                            "augmentation_id"
                        ]
                        == augmentation_id
                    )
                    &
                    (
                        shift_df[
                            "model_config"
                        ]
                        == model_config
                    )
                ].copy()

                seed_means = (
                    subset
                    .groupby(
                        "seed"
                    )[
                        "delta_mse"
                    ]
                    .mean()
                    .reindex(
                        SEEDS
                    )
                )

                if (
                    seed_means
                    .isna()
                    .any()
                ):

                    raise RuntimeError(
                        "Missing seed in H4.1 "
                        f"{shift_type} "
                        f"{augmentation_id} "
                        f"{model_config}"
                    )

                values = (
                    seed_means
                    .to_numpy(
                        dtype=float
                    )
                )

                sign_result = (
                    exact_scalar_signflip(
                        values
                    )
                )

                records.append({

                    "shift_type":
                        shift_type,

                    "augmentation_id":
                        augmentation_id,

                    "model_config":
                        model_config,

                    **sign_result,
                })

    result = pd.DataFrame(
        records
    )

    assert len(result) == 24

    # Correct the 12 A × M tests separately
    # within each shift family.

    frames = []

    for shift_type in SHIFT_TYPES:

        part = result[
            result["shift_type"]
            == shift_type
        ].copy()

        _, fdr_p, _, _ = (
            multipletests(
                part[
                    "exact_signflip_p"
                ],
                alpha=ALPHA,
                method="fdr_bh",
            )
        )

        _, bonf_p, _, _ = (
            multipletests(
                part[
                    "exact_signflip_p"
                ],
                alpha=ALPHA,
                method="bonferroni",
            )
        )

        part[
            "p_fdr_bh"
        ] = fdr_p

        part[
            "p_bonferroni"
        ] = bonf_p

        part[
            "significant_fdr"
        ] = (
            part[
                "p_fdr_bh"
            ]
            < ALPHA
        )

        part[
            "significant_bonferroni"
        ] = (
            part[
                "p_bonferroni"
            ]
            < ALPHA
        )

        frames.append(
            part
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Nested model helper
# ============================================================

def nested_model_test(
    data,
    full_formula,
    reduced_formula,
):

    full = smf.ols(
        full_formula,
        data=data,
    ).fit()

    reduced = smf.ols(
        reduced_formula,
        data=data,
    ).fit()

    df_num = int(
        reduced.df_resid
        - full.df_resid
    )

    if df_num <= 0:

        raise RuntimeError(
            "Reduced model does not "
            "have fewer parameters.\n"
            f"Full df_resid={full.df_resid}\n"
            f"Reduced df_resid={reduced.df_resid}"
        )

    comparison = (
        full.compare_f_test(
            reduced
        )
    )

    return {
        "partial_F":
            float(
                comparison[0]
            ),

        "ordinary_nested_p":
            float(
                comparison[1]
            ),

        "df_num":
            int(
                comparison[2]
            ),

        "df_den":
            int(
                full.df_resid
            ),
    }


# ============================================================
# H4.2 / H4.3 / H4.4
#
# IMPORTANT:
#
# These are secondary nested-model summaries.
#
# We intentionally use a two-way model here:
#
# A + M + V
# + A:M
# + A:V
# + M:V
#
# rather than attempting to test a lower-order term while
# simultaneously retaining the three-way interaction that
# contains that term.
#
# This fixes the nesting problem from the previous script.
# ============================================================

def secondary_interaction_tests(
    data
):

    h42_records = []
    h43_records = []
    h44_records = []

    full_two_way = (
        "delta_mse ~ "
        "C(augmentation_id) "
        "+ C(model_config) "
        "+ severity_z "
        "+ C(augmentation_id):C(model_config) "
        "+ C(augmentation_id):severity_z "
        "+ C(model_config):severity_z"
    )

    # --------------------------------------------------------
    # H4.2:
    # remove A:M
    # --------------------------------------------------------

    reduced_h42 = (
        "delta_mse ~ "
        "C(augmentation_id) "
        "+ C(model_config) "
        "+ severity_z "
        "+ C(augmentation_id):severity_z "
        "+ C(model_config):severity_z"
    )

    # --------------------------------------------------------
    # H4.3:
    # remove A:V
    # --------------------------------------------------------

    reduced_h43 = (
        "delta_mse ~ "
        "C(augmentation_id) "
        "+ C(model_config) "
        "+ severity_z "
        "+ C(augmentation_id):C(model_config) "
        "+ C(model_config):severity_z"
    )

    # --------------------------------------------------------
    # H4.4:
    # remove M:V
    # --------------------------------------------------------

    reduced_h44 = (
        "delta_mse ~ "
        "C(augmentation_id) "
        "+ C(model_config) "
        "+ severity_z "
        "+ C(augmentation_id):C(model_config) "
        "+ C(augmentation_id):severity_z"
    )

    for shift_type in SHIFT_TYPES:

        part = data[
            data["shift_type"]
            == shift_type
        ].copy()

        h42 = nested_model_test(
            part,
            full_two_way,
            reduced_h42,
        )

        h42_records.append({

            "shift_type":
                shift_type,

            "hypothesis":
                "H4.2",

            "effect":
                "A x M",

            **h42,

            "role":
                (
                    "Secondary two-way "
                    "nested-model summary."
                ),
        })

        h43 = nested_model_test(
            part,
            full_two_way,
            reduced_h43,
        )

        h43_records.append({

            "shift_type":
                shift_type,

            "hypothesis":
                "H4.3",

            "effect":
                "A x V",

            **h43,

            "role":
                (
                    "Secondary two-way "
                    "nested-model summary."
                ),
        })

        h44 = nested_model_test(
            part,
            full_two_way,
            reduced_h44,
        )

        h44_records.append({

            "shift_type":
                shift_type,

            "hypothesis":
                "H4.4",

            "effect":
                "M x V",

            **h44,

            "role":
                (
                    "Secondary two-way "
                    "nested-model summary."
                ),
        })

    return (
        pd.DataFrame(
            h42_records
        ),
        pd.DataFrame(
            h43_records
        ),
        pd.DataFrame(
            h44_records
        ),
    )


# ============================================================
# Per-severity A × M sensitivity
#
# Secondary analysis.
#
# At each severity:
#
# full:
#     delta ~ A * M
#
# reduced:
#     delta ~ A + M
#
# Multiplicity correction is performed separately
# inside temperature and humidity.
# ============================================================

def per_severity_AxM(
    data
):

    records = []

    for shift_type in SHIFT_TYPES:

        shift_df = data[
            data["shift_type"]
            == shift_type
        ].copy()

        severities = sorted(
            shift_df[
                "severity"
            ].unique()
        )

        for severity in severities:

            part = shift_df[
                shift_df[
                    "severity"
                ]
                == severity
            ].copy()

            full_formula = (
                "delta_mse ~ "
                "C(augmentation_id) "
                "* C(model_config)"
            )

            reduced_formula = (
                "delta_mse ~ "
                "C(augmentation_id) "
                "+ C(model_config)"
            )

            result = nested_model_test(
                part,
                full_formula,
                reduced_formula,
            )

            shift_value = float(
                part[
                    "shift_value"
                ]
                .iloc[0]
            )

            records.append({

                "shift_type":
                    shift_type,

                "severity":
                    int(
                        severity
                    ),

                "shift_value":
                    shift_value,

                **result,
            })

    result_df = pd.DataFrame(
        records
    )

    frames = []

    for shift_type in SHIFT_TYPES:

        part = result_df[
            result_df[
                "shift_type"
            ]
            == shift_type
        ].copy()

        pvalues = (
            part[
                "ordinary_nested_p"
            ]
            .to_numpy()
        )

        _, fdr_p, _, _ = (
            multipletests(
                pvalues,
                alpha=ALPHA,
                method="fdr_bh",
            )
        )

        _, bonf_p, _, _ = (
            multipletests(
                pvalues,
                alpha=ALPHA,
                method="bonferroni",
            )
        )

        part[
            "p_fdr_bh"
        ] = fdr_p

        part[
            "p_bonferroni"
        ] = bonf_p

        part[
            "significant_fdr"
        ] = (
            part[
                "p_fdr_bh"
            ]
            < ALPHA
        )

        part[
            "significant_bonferroni"
        ] = (
            part[
                "p_bonferroni"
            ]
            < ALPHA
        )

        frames.append(
            part
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print(
        "=" * 78
    )

    print(
        "PHASE 4 FORMAL "
        "INTERACTION ANALYSIS"
    )

    print(
        "=" * 78
    )

    print()

    print(
        "Primary hypothesis:"
    )

    print(
        "H4.5 = Augmentation "
        "x Model x Severity"
    )

    print()

    print(
        "Primary replication unit:"
    )

    print(
        "training seed (n = 10)"
    )

    print()

    print(
        "Primary inference:"
    )

    print(
        "exact seed-level sign-flip test"
    )

    print()

    print(
        "Exact sign patterns per "
        "shift family: 2^10 = 1024"
    )

    print()


    # ========================================================
    # Load
    # ========================================================

    data = load_data()

    data = (
        add_standardized_severity(
            data
        )
    )

    print(
        "EXP1C paired rows:",
        len(data)
    )

    print(
        "Temperature:",
        len(
            data[
                data["shift_type"]
                == "temperature"
            ]
        )
    )

    print(
        "Humidity:",
        len(
            data[
                data["shift_type"]
                == "humidity"
            ]
        )
    )


    # ========================================================
    # PRIMARY H4.5
    # ========================================================

    (
        primary_df,
        seed_contrast_df,
        component_df,
    ) = analyze_H45(
        data
    )

    # SAVE IMMEDIATELY.
    #
    # This means that even if a later secondary analysis
    # fails, the expensive/important primary result is safe.

    primary_df.to_csv(
        PRIMARY_FILE,
        index=False,
    )

    seed_contrast_df.to_csv(
        PRIMARY_CONTRAST_FILE,
        index=False,
    )

    component_df.to_csv(
        PRIMARY_COMPONENT_FILE,
        index=False,
    )

    print()
    print(
        "Primary H4.5 results saved."
    )


    # ========================================================
    # Regression coefficients
    # ========================================================

    coefficient_df = (
        regression_coefficients(
            data
        )
    )

    coefficient_df.to_csv(
        COEFFICIENT_FILE,
        index=False,
    )


    # ========================================================
    # Cluster-robust sensitivity
    # ========================================================

    sensitivity_df = (
        cluster_robust_sensitivity(
            data
        )
    )

    sensitivity_df.to_csv(
        SENSITIVITY_FILE,
        index=False,
    )


    # ========================================================
    # H4.1
    # ========================================================

    h41_df = analyze_H41(
        data
    )

    h41_df.to_csv(
        H41_FILE,
        index=False,
    )


    # ========================================================
    # H4.2 / H4.3 / H4.4
    # ========================================================

    (
        h42_df,
        h43_df,
        h44_df,
    ) = secondary_interaction_tests(
        data
    )

    h42_df.to_csv(
        H42_FILE,
        index=False,
    )

    h43_df.to_csv(
        H43_FILE,
        index=False,
    )

    h44_df.to_csv(
        H44_FILE,
        index=False,
    )


    # ========================================================
    # Per-severity A × M sensitivity
    # ========================================================

    per_severity_df = (
        per_severity_AxM(
            data
        )
    )

    per_severity_df.to_csv(
        PER_SEVERITY_FILE,
        index=False,
    )


    # ========================================================
    # JSON report
    # ========================================================

    report = {

        "phase":
            "PHASE4",

        "analysis":
            "FORMAL_INTERACTIONS",

        "dataset":
            "D00",

        "primary_hypothesis":
            "H4.5",

        "primary_effect":
            "augmentation x model x severity",

        "primary_replication_unit":
            "training_seed",

        "n_seeds":
            10,

        "n_exp1c_pairs":
            1560,

        "temperature_pairs":
            720,

        "humidity_pairs":
            840,

        "primary_method":
            (
                "exact multivariate "
                "seed-level sign-flip test"
            ),

        "exact_sign_patterns":
            1024,

        "interaction_contrasts_per_seed":
            6,

        "reference_augmentation":
            "A1",

        "reference_model":
            "M0",

        "primary_shift_families":
            [
                "temperature",
                "humidity",
            ],

        "sensitivity_analysis":
            (
                "OLS full interaction model "
                "with cluster-robust covariance "
                "by training seed"
            ),

        "cluster_warning":
            (
                "Only 10 seed clusters; "
                "cluster-robust inference is "
                "sensitivity evidence, not "
                "the sole primary test."
            ),

        "secondary_tests":
            {
                "H4.1":
                    (
                        "seed-level augmentation "
                        "effects"
                    ),

                "H4.2":
                    (
                        "A x M two-way "
                        "nested-model summary"
                    ),

                "H4.3":
                    (
                        "A x V two-way "
                        "nested-model summary"
                    ),

                "H4.4":
                    (
                        "M x V two-way "
                        "nested-model summary"
                    ),

                "per_severity_AxM":
                    (
                        "secondary sensitivity "
                        "with FDR and Bonferroni"
                    ),
            },

        "important_interpretation_limits":
            [
                (
                    "D is fixed to D00."
                ),
                (
                    "Model configurations jointly "
                    "change depth, width, and "
                    "parameter count."
                ),
                (
                    "Augmentation is oracle-relabeled "
                    "synthetic feature-space "
                    "augmentation."
                ),
                (
                    "Separate temperature and humidity "
                    "results do not by themselves "
                    "constitute a formal cross-shift "
                    "interaction test."
                ),
                (
                    "Results apply to the controlled "
                    "synthetic CEA setting tested here."
                ),
            ],
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


    # ========================================================
    # Console report
    # ========================================================

    print()
    print(
        "=" * 78
    )

    print(
        "PRIMARY H4.5 RESULTS"
    )

    print(
        "=" * 78
    )

    print()

    print(
        primary_df[
            [
                "shift_type",
                "observed_multivariate_statistic",
                "exact_signflip_p",
                "p_fdr_bh_across_shift_families",
                "p_bonferroni_across_shift_families",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "CLUSTER-ROBUST "
        "SENSITIVITY"
    )

    print(
        "=" * 78
    )

    print()

    print(
        sensitivity_df[
            [
                "shift_type",
                "n_seed_clusters",
                "wald_statistic",
                "cluster_robust_p",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "SECONDARY H4.2"
    )

    print(
        "=" * 78
    )

    print()

    print(
        h42_df.to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "SECONDARY H4.3"
    )

    print(
        "=" * 78
    )

    print()

    print(
        h43_df.to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "SECONDARY H4.4"
    )

    print(
        "=" * 78
    )

    print()

    print(
        h44_df.to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "PER-SEVERITY "
        "A x M SENSITIVITY"
    )

    print(
        "=" * 78
    )

    print()

    print(
        per_severity_df[
            [
                "shift_type",
                "severity",
                "shift_value",
                "partial_F",
                "ordinary_nested_p",
                "p_fdr_bh",
                "p_bonferroni",
            ]
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 78
    )

    print(
        "PHASE 4 FORMAL "
        "ANALYSIS: COMPLETE"
    )

    print(
        "=" * 78
    )

    print()

    print(
        "Results saved in:"
    )

    print(
        OUTPUT_DIR
    )

    print()


if __name__ == "__main__":

    main()