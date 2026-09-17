"""
Phase 2 augmentation operators.

IMPORTANT
---------
This module implements the augmentation protocol frozen in the
Phase 2 preregistration.

A0 = no augmentation
A1 = local environmental Gaussian jitter
A2 = warm-directed interpolation
A3 = humid-directed interpolation
A4 = joint warm-humid interpolation

For A1-A4:
    50% of rows are transformed.
    50% remain original.
    Total dataset size does NOT change.

For transformed rows, target_growth is recomputed using the
same target-generating function used in Phase 1.
"""

import numpy as np
import pandas as pd


TARGET = "target_growth"

ENVIRONMENTAL_FEATURES = [
    "air_temperature",
    "humidity",
    "water_temperature",
    "ph",
    "ec",
    "light",
    "co2",
]

FEATURES = ENVIRONMENTAL_FEATURES + ["growth_stage"]


AUGMENTATION_IDS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]


# ---------------------------------------------------------
# A1 noise scales
#
# Frozen at 0.5 × normal-domain standard deviation.
# ---------------------------------------------------------

A1_NOISE_SCALES = {
    "air_temperature": 0.75,
    "humidity": 2.50,
    "water_temperature": 0.50,
    "ph": 0.075,
    "ec": 0.10,
    "light": 30.0,
    "co2": 40.0,
}


# Source-domain targets.
#
# These values come from the already-defined Phase 1
# source domains, NOT from held-out OOD test results.

WARM_TARGET_TEMPERATURE = 28.0
HUMID_TARGET_HUMIDITY = 82.0


# ---------------------------------------------------------
# Target function
# ---------------------------------------------------------

def deterministic_growth(df):
    """
    Calculate the deterministic component of target_growth.

    This is the SAME target function used in Phase 1.
    """

    t = df["air_temperature"].to_numpy(dtype=float)
    h = df["humidity"].to_numpy(dtype=float)
    tw = df["water_temperature"].to_numpy(dtype=float)
    ph = df["ph"].to_numpy(dtype=float)
    ec = df["ec"].to_numpy(dtype=float)
    light = df["light"].to_numpy(dtype=float)
    co2 = df["co2"].to_numpy(dtype=float)
    stage = df["growth_stage"].to_numpy(dtype=float)

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

    return growth


def recompute_target(df, rng, noise_std=0.25):
    """
    Recompute target_growth after augmentation.

    Fresh observation noise is added using the same
    sigma = 0.25 used in Phase 1.
    """

    deterministic = deterministic_growth(df)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=len(df),
    )

    return deterministic + noise


# ---------------------------------------------------------
# Main augmentation function
# ---------------------------------------------------------

def apply_augmentation(
    df,
    augmentation_id,
    seed,
    fraction=0.50,
):
    """
    Apply one Phase 2 augmentation strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Original training dataset.

    augmentation_id : str
        A0, A1, A2, A3, or A4.

    seed : int
        Augmentation random seed.

    fraction : float
        Fraction of rows transformed.

        Frozen Phase 2 value = 0.50.

    Returns
    -------
    augmented_df : pandas.DataFrame
        Dataset with the SAME number of rows as input.

    metadata : dict
        Information about the augmentation operation.
    """

    if augmentation_id not in AUGMENTATION_IDS:
        raise ValueError(
            f"Unknown augmentation_id: {augmentation_id}. "
            f"Expected one of {AUGMENTATION_IDS}"
        )

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "fraction must be between 0 and 1"
        )

    missing_columns = [
        column
        for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Never modify the caller's original DataFrame.
    output = df.copy(deep=True)

    # -----------------------------------------------------
    # A0
    # -----------------------------------------------------

    if augmentation_id == "A0":

        metadata = {
            "augmentation_id": "A0",
            "seed": seed,
            "fraction": 0.0,
            "n_total": len(output),
            "n_augmented": 0,
            "selected_positions": [],
        }

        return output, metadata

    # Independent RNG.
    rng = np.random.default_rng(seed)

    n_total = len(output)

    n_augmented = int(
        round(n_total * fraction)
    )

    # Select ROW POSITIONS rather than assuming
    # DataFrame index = 0 ... n-1.
    positions = rng.choice(
        n_total,
        size=n_augmented,
        replace=False,
    )

    positions = np.sort(positions)

    selected_index = output.index[positions]

    transformed = output.loc[
        selected_index
    ].copy(deep=True)

    # -----------------------------------------------------
    # A1
    # Local environmental Gaussian jitter
    # -----------------------------------------------------

    if augmentation_id == "A1":

        for feature, scale in A1_NOISE_SCALES.items():

            perturbation = rng.normal(
                loc=0.0,
                scale=scale,
                size=n_augmented,
            )

            transformed[feature] = (
                transformed[feature].to_numpy(dtype=float)
                + perturbation
            )

    # -----------------------------------------------------
    # A2
    # Warm-directed interpolation
    # -----------------------------------------------------

    elif augmentation_id == "A2":

        alpha = rng.uniform(
            0.0,
            1.0,
            size=n_augmented,
        )

        old_temperature = transformed[
            "air_temperature"
        ].to_numpy(dtype=float)

        transformed["air_temperature"] = (
            old_temperature
            + alpha
            * (
                WARM_TARGET_TEMPERATURE
                - old_temperature
            )
        )

    # -----------------------------------------------------
    # A3
    # Humid-directed interpolation
    # -----------------------------------------------------

    elif augmentation_id == "A3":

        alpha = rng.uniform(
            0.0,
            1.0,
            size=n_augmented,
        )

        old_humidity = transformed[
            "humidity"
        ].to_numpy(dtype=float)

        transformed["humidity"] = (
            old_humidity
            + alpha
            * (
                HUMID_TARGET_HUMIDITY
                - old_humidity
            )
        )

    # -----------------------------------------------------
    # A4
    # Joint warm + humid interpolation
    # -----------------------------------------------------

    elif augmentation_id == "A4":

        alpha_temperature = rng.uniform(
            0.0,
            1.0,
            size=n_augmented,
        )

        alpha_humidity = rng.uniform(
            0.0,
            1.0,
            size=n_augmented,
        )

        old_temperature = transformed[
            "air_temperature"
        ].to_numpy(dtype=float)

        old_humidity = transformed[
            "humidity"
        ].to_numpy(dtype=float)

        transformed["air_temperature"] = (
            old_temperature
            + alpha_temperature
            * (
                WARM_TARGET_TEMPERATURE
                - old_temperature
            )
        )

        transformed["humidity"] = (
            old_humidity
            + alpha_humidity
            * (
                HUMID_TARGET_HUMIDITY
                - old_humidity
            )
        )

    # -----------------------------------------------------
    # Recalculate labels
    # -----------------------------------------------------

    transformed[TARGET] = recompute_target(
        transformed,
        rng,
        noise_std=0.25,
    )

    # Put transformed rows back.
    output.loc[
        selected_index,
        transformed.columns,
    ] = transformed

    metadata = {
        "augmentation_id": augmentation_id,
        "seed": seed,
        "fraction": fraction,
        "n_total": n_total,
        "n_augmented": n_augmented,
        "selected_positions": positions.tolist(),
    }

    return output, metadata