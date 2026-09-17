"""
generate_ood_data.py

Experiment 1B
-------------

Generate three held-out synthetic OOD test environments:

1. extreme_warm
2. extreme_humid
3. warm_humid

IMPORTANT:
These datasets are TEST-ONLY datasets.

They must NOT be used:
- to construct D00-D09
- for model training
- for model selection
- for hyperparameter tuning

The purpose is to evaluate how the models from the
Experiment 1 setup generalize to genuinely held-out
environmental distributions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("data/ood_test")

N_SAMPLES = 6000


# ============================================================
# 2. HELD-OUT OOD DOMAIN PARAMETERS
# ============================================================
#
# Format:
#
# "feature": (mean, standard_deviation)
#
# These domains were NOT used in D00-D09.
# ============================================================

OOD_DOMAIN_PARAMS = {

    # --------------------------------------------------------
    # OOD 1: Extreme Warm
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
    # OOD 2: Extreme Humidity
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
    # OOD 3: Warm + Humid
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
# 3. REPRODUCIBLE RANDOM SEEDS
# ============================================================
#
# Feature generation and target noise use different seeds.
# ============================================================

FEATURE_SEEDS = {

    "extreme_warm": 8000,

    "extreme_humid": 8001,

    "warm_humid": 8002,
}


TARGET_SEEDS = {

    "extreme_warm": 9000,

    "extreme_humid": 9001,

    "warm_humid": 9002,
}


# ============================================================
# 4. GENERATE CUSTOM DOMAIN
# ============================================================

def generate_custom_domain(
    domain_name,
    domain_parameters,
    n_samples,
    seed,
    split="ood_test",
):
    """
    Generate synthetic input features X for one OOD domain.

    Parameters
    ----------
    domain_name:
        Name of the OOD environment.

    domain_parameters:
        Dictionary containing mean and standard deviation
        for each environmental feature.

    n_samples:
        Number of observations.

    seed:
        Random seed.

    split:
        Dataset split label.

    Returns
    -------
    pandas.DataFrame
    """

    rng = np.random.default_rng(seed)

    data = {}

    # --------------------------------------------------------
    # Generate environmental features
    # --------------------------------------------------------

    for feature, (mean, std) in domain_parameters.items():

        data[feature] = rng.normal(
            loc=mean,
            scale=std,
            size=n_samples,
        )

    # --------------------------------------------------------
    # Generate growth stage
    #
    # 0 = early stage
    # 1 = late stage
    # --------------------------------------------------------

    data["growth_stage"] = rng.uniform(
        low=0.0,
        high=1.0,
        size=n_samples,
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(data)

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    df["domain"] = domain_name

    df["sample_id"] = [
        f"{split}_{domain_name}_{seed}_{i}"
        for i in range(n_samples)
    ]

    return df


# ============================================================
# 5. ADD TARGET
# ============================================================
#
# IMPORTANT:
#
# We use the SAME target-generating equation used in the
# original synthetic experiment.
#
# The environmental distribution changes, while the target
# mechanism remains fixed.
# ============================================================

def add_target(df, seed):

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Extract input variables
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
    # Synthetic growth function
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
        size=len(df),
    )


    # Don't modify the original DataFrame directly
    df = df.copy()


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    df["target_growth"] = growth + noise

    return df


# ============================================================
# 6. VALIDATE GENERATED DATASET
# ============================================================

def validate_dataset(
    df,
    domain_name,
    expected_samples,
):
    """
    Run basic integrity checks before saving the dataset.
    """

    # Correct number of observations
    if len(df) != expected_samples:

        raise ValueError(
            f"{domain_name}: expected "
            f"{expected_samples} rows, "
            f"but received {len(df)}."
        )


    # Target must exist
    if "target_growth" not in df.columns:

        raise ValueError(
            f"{domain_name}: target_growth column missing."
        )


    # No missing values
    if df.isnull().any().any():

        raise ValueError(
            f"{domain_name}: missing values detected."
        )


    # Correct domain label
    unique_domains = df["domain"].unique()

    if (
        len(unique_domains) != 1
        or unique_domains[0] != domain_name
    ):

        raise ValueError(
            f"{domain_name}: incorrect domain labels."
        )


    # Sample IDs must be unique
    if df["sample_id"].duplicated().any():

        raise ValueError(
            f"{domain_name}: duplicate sample IDs detected."
        )


# ============================================================
# 7. GENERATE ONE COMPLETE OOD DATASET
# ============================================================

def generate_ood_dataset(domain_name):

    print()
    print("=" * 60)
    print(f"Generating: {domain_name}")
    print("=" * 60)


    # Get domain parameters
    parameters = OOD_DOMAIN_PARAMS[domain_name]


    # --------------------------------------------------------
    # Step A: Generate X
    # --------------------------------------------------------

    df = generate_custom_domain(

        domain_name=domain_name,

        domain_parameters=parameters,

        n_samples=N_SAMPLES,

        seed=FEATURE_SEEDS[domain_name],

        split="ood_test",
    )


    # --------------------------------------------------------
    # Step B: Generate Y
    # --------------------------------------------------------

    df = add_target(

        df=df,

        seed=TARGET_SEEDS[domain_name],
    )


    # --------------------------------------------------------
    # Step C: Validate
    # --------------------------------------------------------

    validate_dataset(

        df=df,

        domain_name=domain_name,

        expected_samples=N_SAMPLES,
    )


    return df


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 60)
    print("EXPERIMENT 1B")
    print("HELD-OUT OOD DATA GENERATION")
    print("=" * 60)


    # --------------------------------------------------------
    # Create output directory if necessary
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Generate all OOD environments
    # --------------------------------------------------------

    for domain_name in OOD_DOMAIN_PARAMS:

        df = generate_ood_dataset(
            domain_name
        )


        # ----------------------------------------------------
        # Save CSV
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR
            / f"{domain_name}.csv"
        )

        df.to_csv(
            output_file,
            index=False,
        )


        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print(f"Rows: {len(df)}")

        print(
            f"Saved: {output_file}"
        )

        print()

        print("Observed means:")

        columns = [

            "air_temperature",

            "humidity",

            "water_temperature",

            "ph",

            "ec",

            "light",

            "co2",

            "growth_stage",

            "target_growth",
        ]

        print(
            df[columns]
            .mean()
            .round(3)
        )


    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print("OOD DATA GENERATION COMPLETE")

    print("=" * 60)

    print(
        f"Domains generated: "
        f"{len(OOD_DOMAIN_PARAMS)}"
    )

    print(
        f"Samples per domain: "
        f"{N_SAMPLES}"
    )

    print(
        f"Total OOD samples: "
        f"{len(OOD_DOMAIN_PARAMS) * N_SAMPLES}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()