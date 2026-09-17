from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# PHASE 3 — FORMAL INTERACTION ANALYSIS
#
# Primary estimand:
#
# delta_mse = MSE(M) - MSE(M0)
#
# Primary model, separately for temperature and humidity:
#
# delta_mse ~ C(dataset_id) * C(model_config) * severity_z
#
# Cluster-robust covariance:
# cluster = dataset_id × seed
#
# Primary scientific target:
# D × M × V
#
# IMPORTANT:
# - EXP1C only
# - M1/M2 are paired against M0
# - severity is standardized within shift family
# - linear severity interaction is PRIMARY
# - per-severity D × M is a sensitivity analysis
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "results"
    / "experiment_3"
    / "confirmation"
    / "analysis"
    / "paired_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "experiment_3"
    / "confirmation"
    / "analysis"
    / "interactions"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASETS = [
    f"D{i:02d}"
    for i in range(10)
]

MODELS = [
    "M1",
    "M2",
]

SHIFT_TYPES = [
    "temperature",
    "humidity",
]


# ============================================================
# LOAD + PREPARE
# ============================================================


def load_and_prepare():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded paired records: {len(df)}")

    required = [
        "dataset_id",
        "model_config",
        "seed",
        "experiment",
        "environment",
        "shift_type",
        "severity",
        "shift_value",
        "delta_mse",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}"
        )

    # Formal interaction analysis uses EXP1C only.

    df = df[
        df["experiment"] == "EXP1C"
    ].copy()

    df = df[
        df["shift_type"].isin(
            SHIFT_TYPES
        )
    ].copy()

    assert set(
        df["dataset_id"].unique()
    ) == set(DATASETS)

    assert set(
        df["model_config"].unique()
    ) == set(MODELS)

    # Expected:
    #
    # temperature:
    # 10 D × 2 M × 10 seeds × 6 severity
    # = 1200
    #
    # humidity:
    # 10 D × 2 M × 10 seeds × 7 severity
    # = 1400
    #
    # total = 2600

    assert len(df) == 2600

    counts = (
        df["shift_type"]
        .value_counts()
    )

    assert counts["temperature"] == 1200
    assert counts["humidity"] == 1400

    assert df["delta_mse"].notna().all()

    assert np.isfinite(
        df["delta_mse"]
    ).all()

    # Cluster ID:
    # same D × seed cluster contains repeated severity/model
    # observations sharing the same baseline M0 fit.

    df["cluster_id"] = (
        df["dataset_id"].astype(str)
        + "_seed"
        + df["seed"].astype(str)
    )

    assert df["cluster_id"].nunique() == 100

    # Explicit categorical reference ordering.
    #
    # D00 = reference dataset.
    # M1  = reference nonbaseline model.
    #
    # Note:
    # delta itself is already defined relative to M0.

    df["dataset_id"] = pd.Categorical(
        df["dataset_id"],
        categories=DATASETS,
        ordered=True,
    )

    df["model_config"] = pd.Categorical(
        df["model_config"],
        categories=MODELS,
        ordered=True,
    )

    # Standardize severity SEPARATELY for each shift family.

    df["severity_z"] = np.nan

    for shift_type in SHIFT_TYPES:

        mask = (
            df["shift_type"]
            == shift_type
        )

        x = (
            df.loc[
                mask,
                "severity"
            ]
            .astype(float)
        )

        mean = x.mean()
        std = x.std(ddof=0)

        if std <= 0:
            raise RuntimeError(
                f"Invalid severity SD for {shift_type}"
            )

        df.loc[
            mask,
            "severity_z"
        ] = (
            (x - mean) / std
        )

    assert df["severity_z"].notna().all()

    df.to_csv(
        OUTPUT_DIR
        / "interaction_analysis_input.csv",
        index=False,
    )

    return df


# ============================================================
# CLUSTER-ROBUST FIT
# ============================================================


def fit_clustered_model(data, formula):

    model = smf.ols(
        formula=formula,
        data=data,
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups":
                data["cluster_id"],
            "use_correction":
                True,
        },
    )

    return result


# ============================================================
# COEFFICIENT TABLE
# ============================================================


def coefficient_table(result):

    ci = result.conf_int()

    table = pd.DataFrame({
        "term":
            result.params.index,
        "estimate":
            result.params.values,
        "std_error":
            result.bse.values,
        "statistic":
            result.tvalues.values,
        "p_value":
            result.pvalues.values,
        "ci_low":
            ci.iloc[:, 0].values,
        "ci_high":
            ci.iloc[:, 1].values,
    })

    return table


# ============================================================
# SAFE WALD TEST
# ============================================================


def safe_wald(result, terms, label):

    available = [
        term
        for term in terms
        if term in result.params.index
    ]

    if not available:

        return {
            "test": label,
            "statistic": np.nan,
            "df": 0,
            "p_value": np.nan,
            "n_terms_requested":
                len(terms),
            "n_terms_tested":
                0,
            "status":
                "NO_MATCHING_TERMS",
        }

    hypotheses = [
        f"{term} = 0"
        for term in available
    ]

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            test = result.wald_test(
                hypotheses,
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

        return {
            "test":
                label,
            "statistic":
                statistic,
            "df":
                len(available),
            "p_value":
                p_value,
            "n_terms_requested":
                len(terms),
            "n_terms_tested":
                len(available),
            "status":
                "OK",
        }

    except Exception as exc:

        return {
            "test":
                label,
            "statistic":
                np.nan,
            "df":
                len(available),
            "p_value":
                np.nan,
            "n_terms_requested":
                len(terms),
            "n_terms_tested":
                len(available),
            "status":
                f"FAILED: {exc}",
        }


# ============================================================
# IDENTIFY INTERACTION TERMS
#
# With treatment coding:
#
# D00 = reference D
# M1  = reference model
#
# Therefore:
#
# D×M:
# differences between M2-vs-M1 effects across D
# at severity_z = 0.
#
# M×V:
# model difference in severity slope at D00.
#
# D×V:
# D differences in severity slope at M1.
#
# D×M×V:
# whether the model-difference severity slope
# itself changes across D.
# ============================================================


def get_term_groups(result):

    names = list(
        result.params.index
    )

    dxm = []

    mxv = []

    dxv = []

    dxmxv = []

    for term in names:

        has_d = (
            "C(dataset_id)"
            in term
        )

        has_m = (
            "C(model_config)"
            in term
        )

        has_v = (
            "severity_z"
            in term
        )

        colon_count = (
            term.count(":")
        )

        if (
            has_d
            and has_m
            and not has_v
            and colon_count == 1
        ):
            dxm.append(term)

        elif (
            has_m
            and has_v
            and not has_d
            and colon_count == 1
        ):
            mxv.append(term)

        elif (
            has_d
            and has_v
            and not has_m
            and colon_count == 1
        ):
            dxv.append(term)

        elif (
            has_d
            and has_m
            and has_v
            and colon_count == 2
        ):
            dxmxv.append(term)

    return {
        "D_x_M_at_centered_severity":
            dxm,

        "M_x_V_at_reference_D00":
            mxv,

        "D_x_V_at_reference_M1":
            dxv,

        "D_x_M_x_V_linear":
            dxmxv,
    }


# ============================================================
# PRIMARY ANALYSIS
# ============================================================


def primary_analysis(df):

    all_tests = []

    direction_rows = []

    for shift_type in SHIFT_TYPES:

        print()
        print("=" * 76)
        print(
            f"PRIMARY MODEL: {shift_type.upper()}"
        )
        print("=" * 76)

        data = df[
            df["shift_type"]
            == shift_type
        ].copy()

        formula = (
            "delta_mse ~ "
            "C(dataset_id) "
            "* C(model_config) "
            "* severity_z"
        )

        result = fit_clustered_model(
            data,
            formula,
        )

        # -----------------------------------------------
        # Save full textual summary
        # -----------------------------------------------

        summary_path = (
            OUTPUT_DIR
            / f"{shift_type}_primary_model_summary.txt"
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            f.write(
                result.summary().as_text()
            )

        # -----------------------------------------------
        # Save coefficients
        # -----------------------------------------------

        coef = coefficient_table(
            result
        )

        coef.insert(
            0,
            "shift_type",
            shift_type,
        )

        coef.to_csv(
            OUTPUT_DIR
            / f"{shift_type}_primary_coefficients.csv",
            index=False,
        )

        # -----------------------------------------------
        # Omnibus interaction tests
        # -----------------------------------------------

        term_groups = get_term_groups(
            result
        )

        print(
            "Parameter count:",
            len(result.params)
        )

        print(
            "Clusters:",
            data["cluster_id"].nunique()
        )

        for label, terms in (
            term_groups.items()
        ):

            test = safe_wald(
                result,
                terms,
                label,
            )

            test[
                "shift_type"
            ] = shift_type

            all_tests.append(
                test
            )

            print(
                f"{label}: "
                f"stat={test['statistic']}, "
                f"df={test['df']}, "
                f"p={test['p_value']}"
            )

        # -----------------------------------------------
        # Direction summaries
        #
        # Descriptive cell-specific slopes.
        # These are NOT standalone inferential tests.
        # -----------------------------------------------

        for dataset_id in DATASETS:

            for model_config in MODELS:

                cell = data[
                    (
                        data["dataset_id"]
                        == dataset_id
                    )
                    &
                    (
                        data["model_config"]
                        == model_config
                    )
                ].copy()

                # Mean delta across severity/seed.

                mean_delta = (
                    cell["delta_mse"]
                    .mean()
                )

                # Descriptive slope of delta_mse
                # against standardized severity.

                X = (
                    cell["severity_z"]
                    .to_numpy(
                        dtype=float
                    )
                )

                Y = (
                    cell["delta_mse"]
                    .to_numpy(
                        dtype=float
                    )
                )

                if np.std(X) > 0:

                    slope = np.polyfit(
                        X,
                        Y,
                        1,
                    )[0]

                else:

                    slope = np.nan

                direction_rows.append({

                    "shift_type":
                        shift_type,

                    "dataset_id":
                        dataset_id,

                    "model_config":
                        model_config,

                    "mean_delta_mse":
                        mean_delta,

                    "descriptive_severity_slope":
                        slope,

                    "n":
                        len(cell),
                })

    tests_df = pd.DataFrame(
        all_tests
    )

    # FDR adjustment across the 8 primary omnibus
    # tests is provided as a transparent sensitivity
    # summary.
    #
    # The preregistered scientific interpretation
    # should still focus on the named hypotheses,
    # especially D×M×V.

    valid = (
        tests_df["p_value"]
        .notna()
    )

    tests_df[
        "p_value_fdr_bh"
    ] = np.nan

    if valid.any():

        adjusted = multipletests(
            tests_df.loc[
                valid,
                "p_value"
            ],
            method="fdr_bh",
        )[1]

        tests_df.loc[
            valid,
            "p_value_fdr_bh"
        ] = adjusted

    tests_df.to_csv(
        OUTPUT_DIR
        / "omnibus_interaction_tests.csv",
        index=False,
    )

    directions = pd.DataFrame(
        direction_rows
    )

    directions.to_csv(
        OUTPUT_DIR
        / "interaction_direction_summary.csv",
        index=False,
    )

    return tests_df, directions


# ============================================================
# PER-SEVERITY D × M SENSITIVITY ANALYSIS
#
# At each severity:
#
# delta_mse ~ C(D) * C(M)
#
# Clustered by D × seed.
#
# Main question:
# does M1-vs-M2 difference vary across D
# at this severity?
# ============================================================


def per_severity_analysis(df):

    rows = []

    for shift_type in SHIFT_TYPES:

        shift_df = df[
            df["shift_type"]
            == shift_type
        ].copy()

        severity_levels = sorted(
            shift_df[
                "severity"
            ].unique()
        )

        for severity in severity_levels:

            data = shift_df[
                shift_df["severity"]
                == severity
            ].copy()

            formula = (
                "delta_mse ~ "
                "C(dataset_id) "
                "* C(model_config)"
            )

            result = fit_clustered_model(
                data,
                formula,
            )

            terms = []

            for term in (
                result.params.index
            ):

                if (
                    "C(dataset_id)"
                    in term
                    and
                    "C(model_config)"
                    in term
                    and
                    "severity"
                    not in term
                ):

                    terms.append(
                        term
                    )

            test = safe_wald(
                result,
                terms,
                "D_x_M",
            )

            shift_value = (
                data[
                    "shift_value"
                ]
                .iloc[0]
            )

            rows.append({

                "shift_type":
                    shift_type,

                "severity":
                    severity,

                "shift_value":
                    shift_value,

                "statistic":
                    test[
                        "statistic"
                    ],

                "df":
                    test[
                        "df"
                    ],

                "p_value":
                    test[
                        "p_value"
                    ],

                "n":
                    len(data),

                "clusters":
                    data[
                        "cluster_id"
                    ]
                    .nunique(),

                "status":
                    test[
                        "status"
                    ],
            })

    result_df = pd.DataFrame(
        rows
    )

    # Multiple-testing adjustment across the
    # 13 per-severity D×M sensitivity tests.

    valid = (
        result_df["p_value"]
        .notna()
    )

    result_df[
        "p_value_fdr_bh"
    ] = np.nan

    result_df[
        "p_value_bonferroni"
    ] = np.nan

    if valid.any():

        p = result_df.loc[
            valid,
            "p_value"
        ]

        result_df.loc[
            valid,
            "p_value_fdr_bh"
        ] = multipletests(
            p,
            method="fdr_bh",
        )[1]

        result_df.loc[
            valid,
            "p_value_bonferroni"
        ] = multipletests(
            p,
            method="bonferroni",
        )[1]

    result_df.to_csv(
        OUTPUT_DIR
        / "per_severity_DxM_tests.csv",
        index=False,
    )

    return result_df


# ============================================================
# DESCRIPTIVE CELL SUMMARY
# ============================================================


def cell_summary(df):

    summary = (
        df
        .groupby(
            [
                "shift_type",
                "dataset_id",
                "model_config",
                "severity",
                "shift_value",
            ],
            observed=True,
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

            fraction_better_than_M0=(
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
        / "severity_cell_summary.csv",
        index=False,
    )

    return summary


# ============================================================
# REPORT
# ============================================================


def write_report(
    df,
    tests,
    per_severity,
    directions,
):

    report_path = (
        OUTPUT_DIR
        / "phase3_interaction_report.txt"
    )

    lines = []

    lines.append(
        "PHASE 3 FORMAL INTERACTION ANALYSIS"
    )

    lines.append(
        "=" * 76
    )

    lines.append("")

    lines.append(
        "Primary estimand:"
    )

    lines.append(
        "delta_mse = MSE(model) - MSE(M0)"
    )

    lines.append("")

    lines.append(
        "Primary model:"
    )

    lines.append(
        "delta_mse ~ "
        "C(dataset_id) * "
        "C(model_config) * "
        "severity_z"
    )

    lines.append("")

    lines.append(
        "Temperature and humidity are "
        "analyzed separately."
    )

    lines.append(
        "Covariance is cluster-robust by "
        "dataset_id × seed."
    )

    lines.append(
        f"Total EXP1C paired rows: {len(df)}"
    )

    lines.append(
        f"Clusters: {df['cluster_id'].nunique()}"
    )

    lines.append("")

    lines.append(
        "IMPORTANT CODING INTERPRETATION"
    )

    lines.append(
        "-" * 76
    )

    lines.append(
        "D00 is the reference dataset."
    )

    lines.append(
        "M1 is the reference nonbaseline model."
    )

    lines.append(
        "severity_z is centered within each "
        "shift family."
    )

    lines.append("")

    lines.append(
        "Therefore:"
    )

    lines.append(
        "D×M = interaction at centered severity."
    )

    lines.append(
        "M×V = severity-slope difference "
        "at reference D00."
    )

    lines.append(
        "D×V = dataset differences in severity "
        "slope at reference M1."
    )

    lines.append(
        "D×M×V = whether the model-difference "
        "severity slope varies across datasets."
    )

    lines.append("")

    lines.append(
        "PRIMARY OMNIBUS TESTS"
    )

    lines.append(
        "-" * 76
    )

    lines.append(
        tests.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "PER-SEVERITY D×M SENSITIVITY TESTS"
    )

    lines.append(
        "-" * 76
    )

    lines.append(
        per_severity.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "DIRECTION SUMMARY"
    )

    lines.append(
        "-" * 76
    )

    # Show strongest absolute descriptive
    # severity slopes.

    strongest = (
        directions
        .assign(
            abs_slope=lambda x:
                x[
                    "descriptive_severity_slope"
                ].abs()
        )
        .sort_values(
            "abs_slope",
            ascending=False,
        )
        .head(20)
        .drop(
            columns=[
                "abs_slope"
            ]
        )
    )

    lines.append(
        strongest.to_string(
            index=False
        )
    )

    lines.append("")

    lines.append(
        "INTERPRETATION GUARDRAILS"
    )

    lines.append(
        "-" * 76
    )

    lines.append(
        "1. The linear-severity D×M×V test "
        "is the primary interaction test."
    )

    lines.append(
        "2. Per-severity D×M tests are "
        "sensitivity analyses."
    )

    lines.append(
        "3. Direction-summary slopes are "
        "descriptive, not standalone "
        "inferential tests."
    )

    lines.append(
        "4. Separate temperature/humidity "
        "models do not themselves constitute "
        "a formal cross-shift-type interaction test."
    )

    lines.append(
        "5. Do not describe M2 as universally "
        "superior solely from pooled mean MSE."
    )

    lines.append(
        "6. Model configuration changes depth, "
        "width and parameter count together; "
        "therefore this is not a pure causal "
        "parameter-count experiment."
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(lines)
        )

    return report_path


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 76)
    print("PHASE 3 FORMAL INTERACTION ANALYSIS")
    print("=" * 76)

    df = load_and_prepare()

    print()
    print(
        "EXP1C paired records:",
        len(df)
    )

    print(
        "Temperature:",
        (
            df["shift_type"]
            == "temperature"
        ).sum()
    )

    print(
        "Humidity:",
        (
            df["shift_type"]
            == "humidity"
        ).sum()
    )

    print(
        "Clusters:",
        df["cluster_id"].nunique()
    )

    # Descriptive cells

    cell_summary(
        df
    )

    # Primary clustered models

    tests, directions = (
        primary_analysis(
            df
        )
    )

    # Per-severity sensitivity

    print()
    print(
        "Running per-severity D x M "
        "sensitivity analysis..."
    )

    per_severity = (
        per_severity_analysis(
            df
        )
    )

    report_path = write_report(
        df,
        tests,
        per_severity,
        directions,
    )

    print()
    print("=" * 76)
    print(
        "PHASE 3 INTERACTION ANALYSIS COMPLETE"
    )
    print("=" * 76)

    print()

    print(
        "Primary omnibus tests:"
    )

    print(
        tests[
            [
                "shift_type",
                "test",
                "statistic",
                "df",
                "p_value",
                "p_value_fdr_bh",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Report:\n{report_path}"
    )

    print()

    print(
        "Do not make the final Phase 3 "
        "conclusion until these outputs "
        "have been reviewed."
    )


if __name__ == "__main__":
    main()