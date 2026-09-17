"""
Phase 6 augmentation for generator G2.

A0 = no augmentation
A1 = local environmental jitter
A2 = temperature-directed augmentation
A3 = humidity-directed augmentation
A4 = combined temperature + humidity augmentation

Phase 6 difference:
G2 contains statistical dependencies between variables. Directed
augmentation therefore propagates relevant changes through the frozen
G2 dependency relationships before recomputing the target.

IMPORTANT:
- Final dataset size remains unchanged.
- A1-A4 replace 50% of observations with transformed observations.
- A0 leaves the original dataset unchanged.
- Labels are recomputed using the frozen G2 target function.
- No held-out evaluation data are used.
"""

import numpy as np
import pandas as pd

from src.generator_v2 import compute_growth


# ============================================================
# Constants
# ============================================================

AUGMENTATION_IDS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]

DEFAULT_FRACTION = 0.50

WARM_TARGET = 28.0
HUMID_TARGET = 82.0


# Half-scale local jitter, conceptually inherited from Phase 2.
JITTER_SCALES = {
    "air_temperature": 0.75,
    "humidity": 2.50,
    "water_temperature": 0.50,
    "ph": 0.075,
    "ec": 0.10,
    "light": 30.0,
    "co2": 40.0,
}


# ============================================================
# Helpers
# ============================================================

def _validate_input(df, augmentation_id, fraction):

    if augmentation_id not in AUGMENTATION_IDS:
        raise ValueError(
            f"Unknown augmentation_id: {augmentation_id}"
        )

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "fraction must be between 0 and 1"
        )

    required = {
        "air_temperature",
        "humidity",
        "water_temperature",
        "ph",
        "ec",
        "light",
        "co2",
        "growth_stage",
        "growth",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


def _clip_environment(df):
    """
    Apply the same frozen bounds used by G2 where appropriate.
    """

    df["humidity"] = df["humidity"].clip(
        35.0,
        98.0,
    )

    df["ph"] = df["ph"].clip(
        5.3,
        6.9,
    )

    df["ec"] = df["ec"].clip(
        0.8,
        2.8,
    )

    df["light"] = df["light"].clip(
        100.0,
        850.0,
    )

    df["co2"] = df["co2"].clip(
        350.0,
        1300.0,
    )

    df["growth_stage"] = df["growth_stage"].clip(
        0.0,
        1.0,
    )

    return df


def _propagate_temperature_change(
    transformed,
    old_temperature,
):
    """
    Propagate an imposed air-temperature change through the
    frozen G2 relationships.

    G2:
        water_temperature includes
            0.50 * (air_temperature - 23)

        humidity contains a within-environment negative
        temperature dependency.

    For augmentation we propagate the CHANGE rather than
    regenerate the entire observation. This preserves the
    observation-specific residual structure.
    """

    delta_temperature = (
        transformed["air_temperature"].to_numpy()
        - old_temperature
    )

    transformed["water_temperature"] = (
        transformed["water_temperature"].to_numpy()
        + 0.50 * delta_temperature
    )

    transformed["humidity"] = (
        transformed["humidity"].to_numpy()
        - 0.65 * delta_temperature
    )

    return transformed


def _propagate_stage_dependencies(transformed):
    """
    Re-establish frozen bounds after transformations.

    Growth stage itself is not augmented in A1-A4, so EC/pH
    are not regenerated here. The function exists to keep the
    Phase 6 dependency-handling step explicit.
    """

    return _clip_environment(transformed)


def _relabel(transformed, rng):
    """
    Recompute labels using frozen G2 response function.
    """

    transformed["growth"] = compute_growth(
        air_temperature=transformed[
            "air_temperature"
        ].to_numpy(),

        humidity=transformed[
            "humidity"
        ].to_numpy(),

        water_temperature=transformed[
            "water_temperature"
        ].to_numpy(),

        ph=transformed[
            "ph"
        ].to_numpy(),

        ec=transformed[
            "ec"
        ].to_numpy(),

        light=transformed[
            "light"
        ].to_numpy(),

        co2=transformed[
            "co2"
        ].to_numpy(),

        growth_stage=transformed[
            "growth_stage"
        ].to_numpy(),

        rng=rng,
    )

    return transformed


# ============================================================
# A1
# ============================================================

def _apply_a1(transformed, rng):
    """
    Local environmental jitter.

    All seven environmental predictors are perturbed.
    Growth stage is not perturbed.
    """

    old_temperature = transformed[
        "air_temperature"
    ].to_numpy().copy()

    for column, scale in JITTER_SCALES.items():

        transformed[column] = (
            transformed[column].to_numpy()
            + rng.normal(
                0.0,
                scale,
                len(transformed),
            )
        )

    # A1 already independently perturbs humidity and water
    # temperature, but we additionally propagate the air-temp
    # component of the perturbation through G2.
    transformed = _propagate_temperature_change(
        transformed,
        old_temperature,
    )

    return transformed


# ============================================================
# A2
# ============================================================

def _apply_a2(transformed, rng):
    """
    Temperature-directed augmentation.

    Move selected observations toward 28 C:

        T_new = T + alpha * (28 - T)

    alpha ~ Uniform(0,1)

    The resulting temperature change is propagated through
    G2 water-temperature and humidity dependencies.
    """

    old_temperature = transformed[
        "air_temperature"
    ].to_numpy().copy()

    alpha = rng.uniform(
        0.0,
        1.0,
        len(transformed),
    )

    transformed["air_temperature"] = (
        old_temperature
        + alpha
        * (
            WARM_TARGET
            - old_temperature
        )
    )

    transformed = _propagate_temperature_change(
        transformed,
        old_temperature,
    )

    return transformed


# ============================================================
# A3
# ============================================================

def _apply_a3(transformed, rng):
    """
    Humidity-directed augmentation.

        H_new = H + alpha * (82 - H)

    alpha ~ Uniform(0,1)

    Humidity is a directly shifted G2 variable, so no
    temperature propagation is required.
    """

    old_humidity = transformed[
        "humidity"
    ].to_numpy().copy()

    alpha = rng.uniform(
        0.0,
        1.0,
        len(transformed),
    )

    transformed["humidity"] = (
        old_humidity
        + alpha
        * (
            HUMID_TARGET
            - old_humidity
        )
    )

    return transformed


# ============================================================
# A4
# ============================================================

def _apply_a4(transformed, rng):
    """
    Combined temperature + humidity directed augmentation.

    Temperature and humidity use independent interpolation
    coefficients, matching the conceptual Phase 2 A4 design.

    Temperature propagation is applied first. The explicit
    humidity-directed transformation is then applied to the
    propagated humidity value.
    """

    old_temperature = transformed[
        "air_temperature"
    ].to_numpy().copy()

    alpha_temperature = rng.uniform(
        0.0,
        1.0,
        len(transformed),
    )

    transformed["air_temperature"] = (
        old_temperature
        + alpha_temperature
        * (
            WARM_TARGET
            - old_temperature
        )
    )

    transformed = _propagate_temperature_change(
        transformed,
        old_temperature,
    )

    current_humidity = transformed[
        "humidity"
    ].to_numpy().copy()

    alpha_humidity = rng.uniform(
        0.0,
        1.0,
        len(transformed),
    )

    transformed["humidity"] = (
        current_humidity
        + alpha_humidity
        * (
            HUMID_TARGET
            - current_humidity
        )
    )

    return transformed


# ============================================================
# Public API
# ============================================================

def apply_phase6_augmentation(
    df,
    augmentation_id,
    seed,
    fraction=DEFAULT_FRACTION,
):
    """
    Apply frozen Phase 6 augmentation.

    Returns
    -------
    augmented_df : pandas.DataFrame
        Same number of rows as input.

    metadata : dict
        Information about the augmentation operation.
    """

    _validate_input(
        df,
        augmentation_id,
        fraction,
    )

    original = df.copy(
        deep=True
    )

    n = len(original)

    # --------------------------------------------------------
    # A0
    # --------------------------------------------------------

    if augmentation_id == "A0":

        metadata = {
            "augmentation_id": "A0",
            "augmentation_seed": int(seed),
            "augmentation_fraction": 0.0,
            "n_original": int(n),
            "n_augmented": 0,
            "n_final": int(n),
            "g2_dependency_propagation": False,
        }

        return (
            original.reset_index(drop=True),
            metadata,
        )

    # --------------------------------------------------------
    # RNG
    # --------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    n_augmented = int(
        round(
            n * fraction
        )
    )

    if n_augmented <= 0:

        raise ValueError(
            "Augmentation produced zero transformed rows."
        )

    # Same selected row positions for a given seed,
    # independent of augmentation identity.
    selection_rng = np.random.default_rng(
        seed
    )

    selected_positions = selection_rng.choice(
        n,
        size=n_augmented,
        replace=False,
    )

    selected_positions = np.sort(
        selected_positions
    )

    transformed = (
        original
        .iloc[selected_positions]
        .copy(deep=True)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Apply transformation
    # --------------------------------------------------------

    if augmentation_id == "A1":

        transformed = _apply_a1(
            transformed,
            rng,
        )

    elif augmentation_id == "A2":

        transformed = _apply_a2(
            transformed,
            rng,
        )

    elif augmentation_id == "A3":

        transformed = _apply_a3(
            transformed,
            rng,
        )

    elif augmentation_id == "A4":

        transformed = _apply_a4(
            transformed,
            rng,
        )

    else:
        raise RuntimeError(
            "Unexpected augmentation ID."
        )

    # --------------------------------------------------------
    # Frozen bounds
    # --------------------------------------------------------

    transformed = _propagate_stage_dependencies(
        transformed
    )

    # --------------------------------------------------------
    # Recompute target
    # --------------------------------------------------------

    transformed = _relabel(
        transformed,
        rng,
    )

    # --------------------------------------------------------
    # Fixed-budget replacement
    #
    # Remove selected originals and replace them with their
    # transformed counterparts.
    # --------------------------------------------------------

    keep_mask = np.ones(
        n,
        dtype=bool,
    )

    keep_mask[
        selected_positions
    ] = False

    untouched = (
        original
        .iloc[
            np.where(
                keep_mask
            )[0]
        ]
        .copy(deep=True)
    )

    augmented_df = pd.concat(
        [
            untouched,
            transformed,
        ],
        ignore_index=True,
    )

    # Deterministic final shuffle.
    augmented_df = augmented_df.sample(
        frac=1.0,
        random_state=int(seed),
    ).reset_index(
        drop=True
    )

    if len(augmented_df) != n:
        raise RuntimeError(
            "Fixed-budget augmentation changed dataset size."
        )

    numeric_columns = [
        "air_temperature",
        "humidity",
        "water_temperature",
        "ph",
        "ec",
        "light",
        "co2",
        "growth_stage",
        "growth",
    ]

    if not np.isfinite(
        augmented_df[
            numeric_columns
        ].to_numpy(dtype=float)
    ).all():

        raise RuntimeError(
            "Augmentation produced non-finite values."
        )

    metadata = {
        "augmentation_id":
            augmentation_id,

        "augmentation_seed":
            int(seed),

        "augmentation_fraction":
            float(fraction),

        "n_original":
            int(n),

        "n_augmented":
            int(n_augmented),

        "n_final":
            int(len(augmented_df)),

        "g2_dependency_propagation":
            augmentation_id
            in {"A1", "A2", "A4"},
    }

    return (
        augmented_df,
        metadata,
    )


# Compatibility alias for future runner.
def apply_augmentation(
    df,
    augmentation_id,
    seed,
    fraction=DEFAULT_FRACTION,
):
    return apply_phase6_augmentation(
        df=df,
        augmentation_id=augmentation_id,
        seed=seed,
        fraction=fraction,
    )