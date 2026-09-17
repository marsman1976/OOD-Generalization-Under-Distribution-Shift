"""
Phase 6 G2 Generator Validation
================================

Validates the independently specified Phase 6 synthetic generator G2
BEFORE any machine-learning experiments are performed.

Checks:
1. Directory structure
2. Training dataset structure
3. Reproducibility
4. Intended dependency structure
5. Temperature severity trajectory
6. Humidity severity trajectory
7. Numerical sanity
8. Target sanity
9. Domain differentiation

IMPORTANT:
This script does NOT train or evaluate ML models.
It validates only the frozen G2 data-generating process.
"""

import numpy as np
import pandas as pd

from src.generator_v2 import (
    FEATURES,
    TARGET,
    TRAINING_COMPOSITIONS,
    TEMPERATURE_LEVELS,
    HUMIDITY_LEVELS,
    DATA_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    OOD_DIR,
    SEVERITY_DIR,
    generate_environment,
    generate_all,
)


# ============================================================
# Validation helpers
# ============================================================

checks_passed = 0


def check(condition, message):
    """
    Record a successful validation check or stop immediately
    if the condition fails.
    """

    global checks_passed

    if not condition:
        raise AssertionError(
            f"[FAIL] {message}"
        )

    checks_passed += 1

    print(
        f"[PASS] {message}"
    )


def section(title):
    """
    Print a formatted section header.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Generate frozen G2 data
# ============================================================

section(
    "PHASE 6 G2 GENERATOR VALIDATION"
)

print(
    "Generating frozen Phase 6 datasets..."
)

generate_all()


# ============================================================
# 1. Directory structure
# ============================================================

section(
    "1. DIRECTORY STRUCTURE"
)


for directory in [
    DATA_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR,
    OOD_DIR,
    SEVERITY_DIR,
]:

    check(
        directory.exists(),
        f"Directory exists: {directory}",
    )


# ============================================================
# 2. Training datasets
# ============================================================

section(
    "2. TRAINING DATASETS"
)


expected_columns = (
    FEATURES
    + [
        TARGET,
        "source_domain",
    ]
)


for dataset_id in TRAINING_COMPOSITIONS:

    path = (
        TRAIN_DIR
        / f"{dataset_id}.csv"
    )

    check(
        path.exists(),
        f"{dataset_id}.csv exists",
    )

    df = pd.read_csv(
        path
    )

    check(
        len(df) == 1000,
        f"{dataset_id} has exactly 1000 rows",
    )

    check(
        list(df.columns)
        == expected_columns,
        f"{dataset_id} columns are correct",
    )

    numeric = df[
        FEATURES + [TARGET]
    ]

    check(
        numeric.notna().all().all(),
        f"{dataset_id} contains no missing numeric values",
    )

    check(
        np.isfinite(
            numeric.to_numpy(
                dtype=float
            )
        ).all(),
        f"{dataset_id} contains only finite numeric values",
    )


# ============================================================
# 3. Reproducibility
# ============================================================

section(
    "3. REPRODUCIBILITY"
)


dataset_a = generate_environment(
    n=1000,
    temperature_mean=23.0,
    humidity_mean=68.0,
    seed=12345,
)


dataset_b = generate_environment(
    n=1000,
    temperature_mean=23.0,
    humidity_mean=68.0,
    seed=12345,
)


check(
    dataset_a.equals(
        dataset_b
    ),
    "Same seed generates identical data",
)


dataset_c = generate_environment(
    n=1000,
    temperature_mean=23.0,
    humidity_mean=68.0,
    seed=12346,
)


check(
    not dataset_a.equals(
        dataset_c
    ),
    "Different seed generates different data",
)


# ============================================================
# 4. Dependency structure
# ============================================================

section(
    "4. DEPENDENCY STRUCTURE"
)


normal_path = (
    TEST_DIR
    / "normal.csv"
)


normal = pd.read_csv(
    normal_path
)


correlation_matrix = normal[
    FEATURES
].corr()


temp_water_corr = (
    correlation_matrix.loc[
        "air_temperature",
        "water_temperature",
    ]
)


light_temp_corr = (
    correlation_matrix.loc[
        "light",
        "air_temperature",
    ]
)


light_co2_corr = (
    correlation_matrix.loc[
        "light",
        "co2",
    ]
)


stage_ec_corr = (
    correlation_matrix.loc[
        "growth_stage",
        "ec",
    ]
)


temp_humidity_corr = (
    correlation_matrix.loc[
        "air_temperature",
        "humidity",
    ]
)


print()
print("Observed G2 correlations:")
print()

print(
    "Air temperature <-> water temperature:",
    f"{temp_water_corr:.4f}",
)

print(
    "Light <-> air temperature:",
    f"{light_temp_corr:.4f}",
)

print(
    "Light <-> CO2:",
    f"{light_co2_corr:.4f}",
)

print(
    "Growth stage <-> EC:",
    f"{stage_ec_corr:.4f}",
)

print(
    "Air temperature <-> humidity:",
    f"{temp_humidity_corr:.4f}",
)


# ------------------------------------------------------------
# Dependency checks
# ------------------------------------------------------------

check(
    temp_water_corr > 0.40,
    (
        "Air temperature and water temperature "
        "show intended positive dependency"
    ),
)


check(
    light_temp_corr > 0.10,
    (
        "Light and air temperature show intended "
        "positive dependency"
    ),
)


check(
    light_co2_corr > 0.30,
    (
        "Light and CO2 show intended positive "
        "dependency"
    ),
)


check(
    stage_ec_corr > 0.50,
    (
        "Growth stage and EC show intended "
        "positive dependency"
    ),
)


check(
    temp_humidity_corr < -0.10,
    (
        "Air temperature and humidity show intended "
        "negative within-environment dependency"
    ),
)


# ============================================================
# 5. Temperature severity
# ============================================================

section(
    "5. TEMPERATURE SEVERITY"
)


observed_temp_means = []


for severity, expected_temperature in enumerate(
    TEMPERATURE_LEVELS
):

    path = (
        SEVERITY_DIR
        / f"temp_s{severity}.csv"
    )

    check(
        path.exists(),
        f"temp_s{severity}.csv exists",
    )

    df = pd.read_csv(
        path
    )

    check(
        len(df) == 1000,
        f"temp_s{severity} contains 1000 rows",
    )

    observed_temperature = (
        df[
            "air_temperature"
        ].mean()
    )

    observed_temp_means.append(
        observed_temperature
    )

    print(
        (
            f"Severity {severity}: "
            f"target={expected_temperature:.1f} C, "
            f"observed="
            f"{observed_temperature:.3f} C"
        )
    )


check(
    np.all(
        np.diff(
            observed_temp_means
        ) > 0
    ),
    (
        "Temperature severity means "
        "increase monotonically"
    ),
)


temperature_difference = np.max(
    np.abs(
        np.array(
            observed_temp_means
        )
        -
        np.array(
            TEMPERATURE_LEVELS
        )
    )
)


print(
    "Maximum temperature mean deviation:",
    f"{temperature_difference:.4f}",
)


check(
    temperature_difference < 0.30,
    (
        "Observed temperature means track "
        "the frozen severity values"
    ),
)


# ============================================================
# 6. Humidity severity
# ============================================================

section(
    "6. HUMIDITY SEVERITY"
)


observed_humidity_means = []


for severity, expected_humidity in enumerate(
    HUMIDITY_LEVELS
):

    path = (
        SEVERITY_DIR
        / f"humidity_s{severity}.csv"
    )

    check(
        path.exists(),
        f"humidity_s{severity}.csv exists",
    )

    df = pd.read_csv(
        path
    )

    check(
        len(df) == 1000,
        f"humidity_s{severity} contains 1000 rows",
    )

    observed_humidity = (
        df[
            "humidity"
        ].mean()
    )

    observed_humidity_means.append(
        observed_humidity
    )

    print(
        (
            f"Severity {severity}: "
            f"target={expected_humidity:.1f}%, "
            f"observed="
            f"{observed_humidity:.3f}%"
        )
    )


check(
    np.all(
        np.diff(
            observed_humidity_means
        ) > 0
    ),
    (
        "Humidity severity means "
        "increase monotonically"
    ),
)


humidity_difference = np.max(
    np.abs(
        np.array(
            observed_humidity_means
        )
        -
        np.array(
            HUMIDITY_LEVELS
        )
    )
)


print(
    "Maximum humidity mean deviation:",
    f"{humidity_difference:.4f}",
)


check(
    humidity_difference < 0.40,
    (
        "Observed humidity means track "
        "the frozen severity values"
    ),
)


# ============================================================
# 7. File count and numerical sanity
# ============================================================

section(
    "7. NUMERICAL SANITY"
)


all_files = (
    list(
        TRAIN_DIR.glob(
            "*.csv"
        )
    )
    +
    list(
        VALIDATION_DIR.glob(
            "*.csv"
        )
    )
    +
    list(
        TEST_DIR.glob(
            "*.csv"
        )
    )
    +
    list(
        OOD_DIR.glob(
            "*.csv"
        )
    )
    +
    list(
        SEVERITY_DIR.glob(
            "*.csv"
        )
    )
)


print(
    f"Total CSV files found: {len(all_files)}"
)


# Expected:
#
# 10 training
# 1 validation
# 4 standard test
# 3 OOD
# 6 temperature severity
# 7 humidity severity
#
# Total = 31

check(
    len(all_files) == 31,
    (
        "Exactly 31 Phase 6 CSV "
        "files were generated"
    ),
)


frames = []


for path in all_files:

    df = pd.read_csv(
        path
    )

    frames.append(
        df
    )


all_data = pd.concat(
    frames,
    ignore_index=True,
)


numeric_columns = (
    FEATURES
    + [TARGET]
)


check(
    all_data[
        numeric_columns
    ].notna().all().all(),
    "No missing numeric values exist",
)


check(
    np.isfinite(
        all_data[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all(),
    "All numeric values are finite",
)


# ============================================================
# Generator bounds
# ============================================================

check(
    all_data[
        "humidity"
    ].between(
        35.0,
        98.0,
    ).all(),
    "Humidity remains within [35, 98]",
)


check(
    all_data[
        "ph"
    ].between(
        5.3,
        6.9,
    ).all(),
    "pH remains within [5.3, 6.9]",
)


check(
    all_data[
        "ec"
    ].between(
        0.8,
        2.8,
    ).all(),
    "EC remains within [0.8, 2.8]",
)


check(
    all_data[
        "light"
    ].between(
        100.0,
        850.0,
    ).all(),
    "Light remains within [100, 850]",
)


check(
    all_data[
        "co2"
    ].between(
        350.0,
        1300.0,
    ).all(),
    "CO2 remains within [350, 1300]",
)


check(
    all_data[
        "growth_stage"
    ].between(
        0.0,
        1.0,
    ).all(),
    "Growth stage remains within [0, 1]",
)


# ============================================================
# 8. Target sanity
# ============================================================

section(
    "8. TARGET SANITY"
)


growth = all_data[
    TARGET
]


growth_mean = (
    growth.mean()
)

growth_std = (
    growth.std()
)

growth_min = (
    growth.min()
)

growth_max = (
    growth.max()
)


print(
    f"Growth mean: {growth_mean:.6f}"
)

print(
    f"Growth SD:   {growth_std:.6f}"
)

print(
    f"Growth min:  {growth_min:.6f}"
)

print(
    f"Growth max:  {growth_max:.6f}"
)


check(
    np.isfinite(
        growth_mean
    ),
    "Growth mean is finite",
)


check(
    np.isfinite(
        growth_std
    ),
    "Growth standard deviation is finite",
)


check(
    growth_std > 0.25,
    (
        "Growth target has "
        "non-trivial variation"
    ),
)


check(
    growth.nunique() > 100,
    (
        "Growth target contains "
        "many distinct values"
    ),
)


# ============================================================
# 9. Domain differentiation
# ============================================================

section(
    "9. DOMAIN DIFFERENTIATION"
)


normal_df = pd.read_csv(
    TEST_DIR
    / "normal.csv"
)


warm_df = pd.read_csv(
    TEST_DIR
    / "warm.csv"
)


humid_df = pd.read_csv(
    TEST_DIR
    / "humid.csv"
)


cool_df = pd.read_csv(
    TEST_DIR
    / "cool.csv"
)


normal_temp = (
    normal_df[
        "air_temperature"
    ].mean()
)

warm_temp = (
    warm_df[
        "air_temperature"
    ].mean()
)

cool_temp = (
    cool_df[
        "air_temperature"
    ].mean()
)

normal_humidity = (
    normal_df[
        "humidity"
    ].mean()
)

humid_humidity = (
    humid_df[
        "humidity"
    ].mean()
)


print()
print("Observed domain means:")
print()

print(
    f"Normal temperature: "
    f"{normal_temp:.3f}"
)

print(
    f"Warm temperature:   "
    f"{warm_temp:.3f}"
)

print(
    f"Cool temperature:   "
    f"{cool_temp:.3f}"
)

print(
    f"Normal humidity:     "
    f"{normal_humidity:.3f}"
)

print(
    f"Humid humidity:      "
    f"{humid_humidity:.3f}"
)


check(
    warm_temp
    >
    normal_temp + 3.0,
    (
        "Warm domain is meaningfully "
        "warmer than normal"
    ),
)


check(
    cool_temp
    <
    normal_temp - 3.0,
    (
        "Cool domain is meaningfully "
        "cooler than normal"
    ),
)


check(
    humid_humidity
    >
    normal_humidity + 8.0,
    (
        "Humid domain is meaningfully "
        "more humid than normal"
    ),
)


# ============================================================
# 10. Evaluation dataset structure
# ============================================================

section(
    "10. EVALUATION DATASET STRUCTURE"
)


expected_test_files = [
    "normal.csv",
    "warm.csv",
    "humid.csv",
    "cool.csv",
]


for filename in expected_test_files:

    path = (
        TEST_DIR
        / filename
    )

    check(
        path.exists(),
        f"Test environment exists: {filename}",
    )

    df = pd.read_csv(
        path
    )

    check(
        len(df) == 1000,
        f"{filename} has 1000 rows",
    )


expected_ood_files = [
    "extreme_warm.csv",
    "extreme_humid.csv",
    "warm_humid.csv",
]


for filename in expected_ood_files:

    path = (
        OOD_DIR
        / filename
    )

    check(
        path.exists(),
        f"OOD environment exists: {filename}",
    )

    df = pd.read_csv(
        path
    )

    check(
        len(df) == 1000,
        f"{filename} has 1000 rows",
    )


validation_path = (
    VALIDATION_DIR
    / "normal.csv"
)


check(
    validation_path.exists(),
    "Normal validation dataset exists",
)


validation_df = pd.read_csv(
    validation_path
)


check(
    len(validation_df) == 1000,
    (
        "Normal validation dataset "
        "has 1000 rows"
    ),
)


# ============================================================
# Final result
# ============================================================

section(
    "VALIDATION COMPLETE"
)


print(
    f"Total checks passed: "
    f"{checks_passed}"
)

print()

print(
    "PHASE 6 G2 GENERATOR VALIDATION: PASS"
)

print()

print(
    "This validates:"
)

print(
    "  - generator structure"
)

print(
    "  - reproducibility"
)

print(
    "  - dependency structure"
)

print(
    "  - severity progression"
)

print(
    "  - numerical integrity"
)

print(
    "  - domain differentiation"
)

print()

print(
    "IMPORTANT:"
)

print(
    "No machine-learning performance "
    "was inspected."
)

print(
    "Do not modify G2 based on later "
    "model performance."
)