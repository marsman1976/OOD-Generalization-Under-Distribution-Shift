"""
Phase 6 — Independent Synthetic Generator G2
=============================================

Purpose
-------
Create an independently specified synthetic CEA data-generating process
for Phase 6 replication.

IMPORTANT
---------
This generator is intentionally different from the Phase 1–5 generator G1.

Main changes:
1. Environmental variables are statistically dependent.
2. Air temperature influences water temperature and humidity.
3. Growth stage influences EC and pH.
4. Light and CO2 are statistically related.
5. The target contains nonlinear feature interactions.
6. Temperature/humidity severity shifts propagate through the dependency
   system instead of modifying only one isolated predictor.

This file does NOT train ML models.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "phase6"

TRAIN_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
OOD_DIR = DATA_DIR / "ood_test"
SEVERITY_DIR = DATA_DIR / "severity_test"


# ============================================================
# Feature names
# ============================================================

FEATURES = [
    "air_temperature",
    "humidity",
    "water_temperature",
    "ph",
    "ec",
    "light",
    "co2",
    "growth_stage",
]

TARGET = "growth"


# ============================================================
# Domain definitions
# ============================================================
#
# These define the central environmental condition.
#
# Other variables are generated conditionally from these
# variables, rather than independently.
# ============================================================

DOMAIN_CONFIGS = {

    "normal": {
        "temperature_mean": 23.0,
        "humidity_mean": 68.0,
    },

    "warm": {
        "temperature_mean": 28.0,
        "humidity_mean": 68.0,
    },

    "humid": {
        "temperature_mean": 23.0,
        "humidity_mean": 82.0,
    },

    "cool": {
        "temperature_mean": 18.0,
        "humidity_mean": 68.0,
    },

    "extreme_warm": {
        "temperature_mean": 33.0,
        "humidity_mean": 68.0,
    },

    "extreme_humid": {
        "temperature_mean": 23.0,
        "humidity_mean": 92.0,
    },

    "warm_humid": {
        "temperature_mean": 30.0,
        "humidity_mean": 88.0,
    },
}


# ============================================================
# Training composition definitions
# ============================================================

TRAINING_COMPOSITIONS = {

    "D00": {
        "normal": 1.00,
        "warm": 0.00,
        "humid": 0.00,
        "cool": 0.00,
    },

    "D01": {
        "normal": 0.90,
        "warm": 0.10,
        "humid": 0.00,
        "cool": 0.00,
    },

    "D02": {
        "normal": 0.80,
        "warm": 0.10,
        "humid": 0.10,
        "cool": 0.00,
    },

    "D03": {
        "normal": 0.70,
        "warm": 0.10,
        "humid": 0.10,
        "cool": 0.10,
    },

    "D04": {
        "normal": 0.60,
        "warm": 0.20,
        "humid": 0.10,
        "cool": 0.10,
    },

    "D05": {
        "normal": 0.50,
        "warm": 0.20,
        "humid": 0.20,
        "cool": 0.10,
    },

    "D06": {
        "normal": 0.50,
        "warm": 0.10,
        "humid": 0.20,
        "cool": 0.20,
    },

    "D07": {
        "normal": 0.40,
        "warm": 0.20,
        "humid": 0.20,
        "cool": 0.20,
    },

    "D08": {
        "normal": 0.30,
        "warm": 0.30,
        "humid": 0.20,
        "cool": 0.20,
    },

    "D09": {
        "normal": 0.25,
        "warm": 0.25,
        "humid": 0.25,
        "cool": 0.25,
    },
}


# ============================================================
# Severity definitions
# ============================================================

TEMPERATURE_LEVELS = [
    23.0,
    25.0,
    27.0,
    29.0,
    31.0,
    33.0,
]

HUMIDITY_LEVELS = [
    65.0,
    70.0,
    75.0,
    80.0,
    85.0,
    90.0,
    92.0,
]


# ============================================================
# G2 target function
# ============================================================

def compute_growth(
    air_temperature,
    humidity,
    water_temperature,
    ph,
    ec,
    light,
    co2,
    growth_stage,
    rng,
):
    """
    Phase 6 G2 response mechanism.

    This differs intentionally from G1.

    Includes:
    - nonlinear environmental responses
    - temperature x humidity interaction
    - light x CO2 interaction
    - EC x growth-stage interaction
    - high-temperature stress
    - high-humidity stress
    """

    # ------------------------------------------
    # Basic environmental response
    # ------------------------------------------

    growth = (
        5.5
        - 0.055 * (air_temperature - 24.0) ** 2
        - 0.006 * (humidity - 70.0) ** 2
        - 0.075 * (water_temperature - 21.0) ** 2
        - 1.50 * (ph - 6.1) ** 2
        - 0.80 * (ec - 1.8) ** 2
    )

    # ------------------------------------------
    # Light response with diminishing returns
    # ------------------------------------------

    growth += (
        0.55
        * np.log1p(light)
        / np.log(801.0)
    )

    # ------------------------------------------
    # CO2 response
    # ------------------------------------------

    growth += (
        0.40
        * np.log1p(co2)
        / np.log(1201.0)
    )

    # ------------------------------------------
    # Growth stage
    # ------------------------------------------

    growth += (
        1.30
        * growth_stage
    )

    # ------------------------------------------
    # Temperature x humidity interaction
    # ------------------------------------------

    growth += (
        -0.0035
        * (air_temperature - 24.0)
        * (humidity - 70.0)
    )

    # ------------------------------------------
    # Light x CO2 interaction
    # ------------------------------------------

    light_scaled = (
        np.log1p(light)
        / np.log(801.0)
    )

    co2_scaled = (
        np.log1p(co2)
        / np.log(1201.0)
    )

    growth += (
        0.35
        * light_scaled
        * co2_scaled
    )

    # ------------------------------------------
    # EC x growth-stage interaction
    # ------------------------------------------

    growth += (
        0.35
        * (ec - 1.7)
        * growth_stage
    )

    # ------------------------------------------
    # High-temperature stress
    # ------------------------------------------

    temperature_stress = np.maximum(
        air_temperature - 29.0,
        0.0,
    )

    growth -= (
        0.08
        * temperature_stress ** 2
    )

    # ------------------------------------------
    # High-humidity stress
    # ------------------------------------------

    humidity_stress = np.maximum(
        humidity - 85.0,
        0.0,
    )

    growth -= (
        0.012
        * humidity_stress ** 2
    )

    # ------------------------------------------
    # Observation noise
    # ------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=0.30,
        size=len(np.atleast_1d(air_temperature)),
    )

    growth = growth + noise

    return growth


# ============================================================
# Core G2 generator
# ============================================================

def generate_environment(
    n,
    temperature_mean,
    humidity_mean,
    seed,
):
    """
    Generate one G2 environment.

    Important:
    Variables are NOT generated independently.

    Dependency structure:

        Light --------> Air temperature
                           |
                           +------> Water temperature
                           |
                           +------> Humidity

        Growth stage --> EC
        Growth stage --> pH

        Light <-------> CO2

    This is a statistical dependency structure.
    It should not be interpreted as a validated biological
    causal model.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------
    # Growth stage
    # ------------------------------------------

    growth_stage = rng.uniform(
        0.0,
        1.0,
        n,
    )

    # ------------------------------------------
    # Latent daylight intensity
    # ------------------------------------------

    daylight = rng.normal(
        0.0,
        1.0,
        n,
    )

    # ------------------------------------------
    # Light
    # ------------------------------------------

    light = (
        500.0
        + 100.0 * daylight
        + rng.normal(
            0.0,
            45.0,
            n,
        )
    )

    light = np.clip(
        light,
        100.0,
        850.0,
    )

    # ------------------------------------------
    # Air temperature
    #
    # Light produces modest correlated variation.
    # ------------------------------------------

    air_temperature = (
        temperature_mean
        + 0.45 * daylight
        + rng.normal(
            0.0,
            1.25,
            n,
        )
    )

    # ------------------------------------------
    # Humidity
    #
    # humidity_mean defines environment.
    #
    # Within an environment, hotter observations
    # tend to have somewhat lower RH.
    # ------------------------------------------

    humidity = (
        humidity_mean
        - 0.65
        * (
            air_temperature
            - temperature_mean
        )
        + rng.normal(
            0.0,
            3.5,
            n,
        )
    )

    humidity = np.clip(
        humidity,
        35.0,
        98.0,
    )

    # ------------------------------------------
    # Water temperature depends on air temp
    # ------------------------------------------

    water_temperature = (
        20.5
        + 0.50
        * (
            air_temperature
            - 23.0
        )
        + rng.normal(
            0.0,
            0.75,
            n,
        )
    )

    # ------------------------------------------
    # EC depends partly on growth stage
    # ------------------------------------------

    ec = (
        1.45
        + 0.45 * growth_stage
        + rng.normal(
            0.0,
            0.12,
            n,
        )
    )

    ec = np.clip(
        ec,
        0.8,
        2.8,
    )

    # ------------------------------------------
    # pH depends weakly on stage and EC
    # ------------------------------------------

    ph = (
        6.15
        - 0.10 * growth_stage
        - 0.08 * (ec - 1.7)
        + rng.normal(
            0.0,
            0.10,
            n,
        )
    )

    ph = np.clip(
        ph,
        5.3,
        6.9,
    )

    # ------------------------------------------
    # CO2 related to daylight/light conditions
    # ------------------------------------------

    co2 = (
        750.0
        + 65.0 * daylight
        + rng.normal(
            0.0,
            70.0,
            n,
        )
    )

    co2 = np.clip(
        co2,
        350.0,
        1300.0,
    )

    # ------------------------------------------
    # Target
    # ------------------------------------------

    growth = compute_growth(
        air_temperature=air_temperature,
        humidity=humidity,
        water_temperature=water_temperature,
        ph=ph,
        ec=ec,
        light=light,
        co2=co2,
        growth_stage=growth_stage,
        rng=rng,
    )

    df = pd.DataFrame({

        "air_temperature":
            air_temperature,

        "humidity":
            humidity,

        "water_temperature":
            water_temperature,

        "ph":
            ph,

        "ec":
            ec,

        "light":
            light,

        "co2":
            co2,

        "growth_stage":
            growth_stage,

        "growth":
            growth,
    })

    return df


# ============================================================
# Named domain generator
# ============================================================

def generate_domain(
    domain_name,
    n,
    seed,
):
    """
    Generate one named Phase 6 domain.
    """

    if domain_name not in DOMAIN_CONFIGS:

        raise ValueError(
            f"Unknown domain: {domain_name}"
        )

    config = DOMAIN_CONFIGS[
        domain_name
    ]

    return generate_environment(
        n=n,
        temperature_mean=config[
            "temperature_mean"
        ],
        humidity_mean=config[
            "humidity_mean"
        ],
        seed=seed,
    )


# ============================================================
# Training composition generator
# ============================================================

def generate_training_dataset(
    dataset_id,
    n=1000,
    seed=0,
):
    """
    Generate D00-D09 using the frozen mixture weights.
    """

    if dataset_id not in TRAINING_COMPOSITIONS:

        raise ValueError(
            f"Unknown dataset_id: {dataset_id}"
        )

    composition = TRAINING_COMPOSITIONS[
        dataset_id
    ]

    rng = np.random.default_rng(seed)

    domains = [
        "normal",
        "warm",
        "humid",
        "cool",
    ]

    probabilities = np.array(
        [
            composition[d]
            for d in domains
        ],
        dtype=float,
    )

    require_sum = probabilities.sum()

    if not np.isclose(
        require_sum,
        1.0,
    ):

        raise ValueError(
            (
                f"{dataset_id} composition "
                f"sums to {require_sum}"
            )
        )

    counts = rng.multinomial(
        n,
        probabilities,
    )

    frames = []

    for i, (
        domain_name,
        count,
    ) in enumerate(
        zip(
            domains,
            counts,
        )
    ):

        if count == 0:
            continue

        domain_seed = (
            seed * 100
            + i
            + 1
        )

        frame = generate_domain(
            domain_name=domain_name,
            n=count,
            seed=domain_seed,
        )

        frame["source_domain"] = (
            domain_name
        )

        frames.append(
            frame
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    # Shuffle the mixture.
    result = result.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    if len(result) != n:

        raise RuntimeError(
            (
                f"{dataset_id}: expected {n} rows, "
                f"generated {len(result)}"
            )
        )

    return result


# ============================================================
# Generate severity environment
# ============================================================

def generate_temperature_severity(
    temperature,
    n,
    seed,
):
    """
    Generate temperature-shift environment.

    Humidity baseline remains centered at 68%, but the
    dependency equations allow temperature-related
    within-environment covariation.
    """

    return generate_environment(
        n=n,
        temperature_mean=float(
            temperature
        ),
        humidity_mean=68.0,
        seed=seed,
    )


def generate_humidity_severity(
    humidity,
    n,
    seed,
):
    """
    Generate humidity-shift environment.
    """

    return generate_environment(
        n=n,
        temperature_mean=23.0,
        humidity_mean=float(
            humidity
        ),
        seed=seed,
    )


# ============================================================
# Directory creation
# ============================================================

def create_directories():

    for directory in [
        TRAIN_DIR,
        VALIDATION_DIR,
        TEST_DIR,
        OOD_DIR,
        SEVERITY_DIR,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Generate complete Phase 6 base data
# ============================================================

def generate_all(
    base_seed=6000,
):
    """
    Generate the complete frozen Phase 6 G2 dataset collection.

    Training:
        D00-D09, n=1000 each

    Validation:
        normal, n=1000

    Standard test:
        normal, warm, humid, cool

    OOD:
        extreme_warm
        extreme_humid
        warm_humid

    Severity:
        temperature: 23,25,27,29,31,33
        humidity: 65,70,75,80,85,90,92

    Each evaluation environment contains 1000 observations.
    """

    create_directories()

    # ------------------------------------------
    # Training datasets
    # ------------------------------------------

    for i, dataset_id in enumerate(
        TRAINING_COMPOSITIONS
    ):

        df = generate_training_dataset(
            dataset_id=dataset_id,
            n=1000,
            seed=base_seed + i,
        )

        df.to_csv(
            TRAIN_DIR
            / f"{dataset_id}.csv",
            index=False,
        )

    # ------------------------------------------
    # Validation
    # ------------------------------------------

    validation_df = generate_domain(
        domain_name="normal",
        n=1000,
        seed=base_seed + 100,
    )

    validation_df.to_csv(
        VALIDATION_DIR
        / "normal.csv",
        index=False,
    )

    # ------------------------------------------
    # Standard tests
    # ------------------------------------------

    standard_domains = [
        "normal",
        "warm",
        "humid",
        "cool",
    ]

    for i, domain_name in enumerate(
        standard_domains
    ):

        df = generate_domain(
            domain_name=domain_name,
            n=1000,
            seed=base_seed + 200 + i,
        )

        df.to_csv(
            TEST_DIR
            / f"{domain_name}.csv",
            index=False,
        )

    # ------------------------------------------
    # OOD tests
    # ------------------------------------------

    ood_domains = [
        "extreme_warm",
        "extreme_humid",
        "warm_humid",
    ]

    for i, domain_name in enumerate(
        ood_domains
    ):

        df = generate_domain(
            domain_name=domain_name,
            n=1000,
            seed=base_seed + 300 + i,
        )

        df.to_csv(
            OOD_DIR
            / f"{domain_name}.csv",
            index=False,
        )

    # ------------------------------------------
    # Temperature severity
    # ------------------------------------------

    for severity, temperature in enumerate(
        TEMPERATURE_LEVELS
    ):

        df = generate_temperature_severity(
            temperature=temperature,
            n=1000,
            seed=base_seed + 400 + severity,
        )

        df.to_csv(
            SEVERITY_DIR
            / f"temp_s{severity}.csv",
            index=False,
        )

    # ------------------------------------------
    # Humidity severity
    # ------------------------------------------

    for severity, humidity in enumerate(
        HUMIDITY_LEVELS
    ):

        df = generate_humidity_severity(
            humidity=humidity,
            n=1000,
            seed=base_seed + 500 + severity,
        )

        df.to_csv(
            SEVERITY_DIR
            / f"humidity_s{severity}.csv",
            index=False,
        )

    print(
        "Phase 6 G2 datasets generated successfully."
    )

    print(
        f"Output directory: {DATA_DIR}"
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    generate_all()