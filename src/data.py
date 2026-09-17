"""
src/data.py

Synthetic Controlled Environment Agriculture (CEA) data utilities.

Experiment 1:
    Original domains:
        - normal
        - warm
        - humid
        - cool

Experiment 1B:
    Held-out OOD domains:
        - extreme_warm
        - extreme_humid
        - warm_humid
"""

import numpy as np
import pandas as pd


# ============================================================
# FEATURES
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

TARGET = "target_growth"


# ============================================================
# ORIGINAL DOMAIN PARAMETERS
# ============================================================
#
# Format:
#
# feature: (mean, standard_deviation)
#
# These are the domains used to construct D00-D09.
# ============================================================

DOMAIN_PARAMS = {

    "normal": {
        "air_temperature": (23.0, 1.5),
        "humidity": (65.0, 5.0),
        "water_temperature": (20.0, 1.0),
        "ph": (6.0, 0.15),
        "ec": (1.7, 0.20),
        "light": (450.0, 60.0),
        "co2": (750.0, 80.0),
    },

    "warm": {
        "air_temperature": (28.0, 1.5),
        "humidity": (62.0, 5.0),
        "water_temperature": (23.0, 1.0),
        "ph": (6.1, 0.15),
        "ec": (1.8, 0.20),
        "light": (500.0, 60.0),
        "co2": (800.0, 80.0),
    },

    "humid": {
        "air_temperature": (23.0, 1.5),
        "humidity": (82.0, 4.0),
        "water_temperature": (20.5, 1.0),
        "ph": (6.0, 0.15),
        "ec": (1.7, 0.20),
        "light": (440.0, 60.0),
        "co2": (740.0, 80.0),
    },

    "cool": {
        "air_temperature": (18.0, 1.5),
        "humidity": (66.0, 5.0),
        "water_temperature": (17.0, 1.0),
        "ph": (5.9, 0.15),
        "ec": (1.6, 0.20),
        "light": (420.0, 60.0),
        "co2": (700.0, 80.0),
    },
}


# ============================================================
# HELD-OUT OOD DOMAIN PARAMETERS
# ============================================================
#
# IMPORTANT:
#
# These domains must NOT be used to construct D00-D09.
#
# They exist only for Experiment 1B OOD evaluation.
# ============================================================

OOD_DOMAIN_PARAMS = {

    # --------------------------------------------------------
    # S1: Extreme Warm
    # --------------------------------------------------------

    "extreme_warm": {
        "air_temperature": (33.0, 1.5),
        "humidity": (58.0, 5.0),
        "water_temperature": (27.0, 1.2),
        "ph": (6.15, 0.15),
        "ec": (1.9, 0.20),
        "light": (540.0, 60.0),
        "co2": (830.0, 80.0),
    },

    # --------------------------------------------------------
    # S2: Extreme Humidity
    # --------------------------------------------------------

    "extreme_humid": {
        "air_temperature": (24.0, 1.5),
        "humidity": (92.0, 3.0),
        "water_temperature": (21.0, 1.0),
        "ph": (6.0, 0.15),
        "ec": (1.7, 0.20),
        "light": (420.0, 60.0),
        "co2": (730.0, 80.0),
    },

    # --------------------------------------------------------
    # S3: Combined Warm + Humid
    # --------------------------------------------------------

    "warm_humid": {
        "air_temperature": (31.0, 1.5),
        "humidity": (88.0, 3.5),
        "water_temperature": (25.0, 1.0),
        "ph": (6.1, 0.15),
        "ec": (1.85, 0.20),
        "light": (500.0, 60.0),
        "co2": (810.0, 80.0),
    },
}


# ============================================================
# TRAINING-DATA COMPOSITIONS
# ============================================================
#
# These define D00-D09.
#
# Each composition sums to 1.0.
# ============================================================

COMPOSITIONS = {

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
# ORIGINAL DOMAIN GENERATOR
# ============================================================

def generate_domain(
    domain_name,
    n_samples,
    seed,
    split="train"
):
    """
    Generate samples from one of the original domains.

    Example:

        generate_domain(
            domain_name="warm",
            n_samples=1000,
            seed=42
        )
    """

    if domain_name not in DOMAIN_PARAMS:
        raise ValueError(
            f"Unknown domain: {domain_name}. "
            f"Available domains: {list(DOMAIN_PARAMS.keys())}"
        )

    return generate_custom_domain(
        domain_name=domain_name,
        domain_parameters=DOMAIN_PARAMS[domain_name],
        n_samples=n_samples,
        seed=seed,
        split=split,
    )


# ============================================================
# CUSTOM DOMAIN GENERATOR
# ============================================================

def generate_custom_domain(
    domain_name,
    domain_parameters,
    n_samples,
    seed,
    split
):
    """
    Generate synthetic feature data using arbitrary domain
    parameters.

    This is particularly useful for generating held-out OOD
    environments.

    The function generates X only.

    target_growth is added separately by add_target().
    """

    # --------------------------------------------------------
    # Reproducible random-number generator
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    data = {}

    # --------------------------------------------------------
    # Generate environmental variables
    # --------------------------------------------------------

    for feature, (mean, std) in domain_parameters.items():

        data[feature] = rng.normal(
            loc=mean,
            scale=std,
            size=n_samples
        )

    # --------------------------------------------------------
    # Generate growth stage
    #
    # 0 = early
    # 1 = late
    # --------------------------------------------------------

    data["growth_stage"] = rng.uniform(
        low=0.0,
        high=1.0,
        size=n_samples
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(data)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df["domain"] = domain_name

    df["sample_id"] = [
        f"{split}_{domain_name}_{seed}_{i}"
        for i in range(n_samples)
    ]

    return df


# ============================================================
# TARGET-GENERATION FUNCTION
# ============================================================

def add_target(
    df,
    seed
):
    """
    Calculate target_growth using the same underlying target
    mechanism for all domains.

    Keeping the target mechanism fixed while changing P(X)
    gives us a controlled covariate-shift experiment.
    """

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Extract variables
    # --------------------------------------------------------

    t = df["air_temperature"].to_numpy()

    h = df["humidity"].to_numpy()

    tw = df["water_temperature"].to_numpy()

    ph = df["ph"].to_numpy()

    ec = df["ec"].to_numpy()

    light = df["light"].to_numpy()

    co2 = df["co2"].to_numpy()

    stage = df["growth_stage"].to_numpy()

    # --------------------------------------------------------
    # Underlying synthetic growth mechanism
    # --------------------------------------------------------

    growth = (
        5.0
        - 0.08 * (t - 23.0) ** 2
        - 0.01 * (h - 68.0) ** 2
        - 0.10 * (tw - 20.0) ** 2
        - 2.0 * (ph - 6.0) ** 2
        - 1.2 * (ec - 1.7) ** 2
        + 0.003 * light
        + 0.0015 * co2
        + 2.0 * stage
        + 0.015 * t * stage
    )

    # --------------------------------------------------------
    # Observation noise
    # --------------------------------------------------------

    noise = rng.normal(
        loc=0.0,
        scale=0.25,
        size=len(df)
    )

    df = df.copy()

    df[TARGET] = growth + noise

    return df


# ============================================================
# TRAINING DATASET GENERATOR
# ============================================================

def create_training_dataset(
    dataset_id,
    n_samples=1000,
    seed=42
):
    """
    Construct one of D00-D09 according to COMPOSITIONS.

    Example:

        df = create_training_dataset(
            "D05",
            n_samples=1000,
            seed=42
        )
    """

    if dataset_id not in COMPOSITIONS:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. "
            f"Available datasets: {list(COMPOSITIONS.keys())}"
        )

    composition = COMPOSITIONS[dataset_id]

    frames = []

    # Used to avoid identical random streams for each domain
    domain_offset = 0

    for domain_name, proportion in composition.items():

        if proportion <= 0:
            continue

        domain_n = int(
            round(n_samples * proportion)
        )

        domain_seed = (
            seed
            + domain_offset
        )

        df_domain = generate_domain(
            domain_name=domain_name,
            n_samples=domain_n,
            seed=domain_seed,
            split="train"
        )

        df_domain = add_target(
            df_domain,
            seed=domain_seed + 10000
        )

        frames.append(
            df_domain
        )

        domain_offset += 1

    df = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Shuffle observations
    # --------------------------------------------------------

    df = df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(
        drop=True
    )

    return df