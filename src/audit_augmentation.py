import numpy as np
import pandas as pd

from src.augmentation import (
    apply_augmentation,
    deterministic_growth,
    A1_NOISE_SCALES,
    WARM_TARGET_TEMPERATURE,
    HUMID_TARGET_HUMIDITY,
)

DATASET = "data/training/D00.csv"
SEED = 11

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

ENV_FEATURES = [
    "air_temperature",
    "humidity",
    "water_temperature",
    "ph",
    "ec",
    "light",
    "co2",
]


def changed_rows(original, augmented, column):
    return ~np.isclose(
        original[column].to_numpy(dtype=float),
        augmented[column].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def main():

    print("=" * 72)
    print("PHASE 2 STRONG AUGMENTATION AUDIT")
    print("=" * 72)

    original = pd.read_csv(DATASET)

    results = {}

    for augmentation_id in ["A1", "A2", "A3", "A4"]:

        augmented, metadata = apply_augmentation(
            original,
            augmentation_id,
            seed=SEED,
            fraction=0.50,
        )

        results[augmentation_id] = (
            augmented,
            metadata,
        )

    # --------------------------------------------------
    # 1. SAME SELECTED POSITIONS
    # --------------------------------------------------

    reference_positions = (
        results["A1"][1]["selected_positions"]
    )

    for augmentation_id in ["A2", "A3", "A4"]:

        assert (
            results[augmentation_id][1]["selected_positions"]
            == reference_positions
        )

    print("1. Same selected rows across A1-A4: PASS")

    assert len(reference_positions) == 500

    selected_mask = np.zeros(
        len(original),
        dtype=bool,
    )

    selected_mask[reference_positions] = True

    unselected_mask = ~selected_mask

    # --------------------------------------------------
    # 2. UNSELECTED ROWS MUST BE EXACTLY UNCHANGED
    # --------------------------------------------------

    for augmentation_id in ["A1", "A2", "A3", "A4"]:

        augmented = results[augmentation_id][0]

        pd.testing.assert_frame_equal(
            original.iloc[unselected_mask],
            augmented.iloc[unselected_mask],
        )

    print("2. Unselected 50% exactly unchanged: PASS")

    # --------------------------------------------------
    # 3. A2 COLUMN-SPECIFIC TRANSFORMATION
    # --------------------------------------------------

    a2 = results["A2"][0]

    for feature in FEATURES:

        mask = changed_rows(
            original,
            a2,
            feature,
        )

        if feature == "air_temperature":

            assert mask.sum() == 500

        else:

            assert mask.sum() == 0

    print("3. A2 changes only air_temperature predictors: PASS")

    # A2 must move selected temperatures toward 28.
    old_t = original.iloc[
        reference_positions
    ]["air_temperature"].to_numpy()

    new_t = a2.iloc[
        reference_positions
    ]["air_temperature"].to_numpy()

    old_distance = np.abs(
        WARM_TARGET_TEMPERATURE - old_t
    )

    new_distance = np.abs(
        WARM_TARGET_TEMPERATURE - new_t
    )

    assert np.all(
        new_distance <= old_distance + 1e-12
    )

    print("4. A2 moves temperature toward 28 C: PASS")

    # --------------------------------------------------
    # 4. A3 COLUMN-SPECIFIC TRANSFORMATION
    # --------------------------------------------------

    a3 = results["A3"][0]

    for feature in FEATURES:

        mask = changed_rows(
            original,
            a3,
            feature,
        )

        if feature == "humidity":

            assert mask.sum() == 500

        else:

            assert mask.sum() == 0

    print("5. A3 changes only humidity predictors: PASS")

    old_h = original.iloc[
        reference_positions
    ]["humidity"].to_numpy()

    new_h = a3.iloc[
        reference_positions
    ]["humidity"].to_numpy()

    old_distance = np.abs(
        HUMID_TARGET_HUMIDITY - old_h
    )

    new_distance = np.abs(
        HUMID_TARGET_HUMIDITY - new_h
    )

    assert np.all(
        new_distance <= old_distance + 1e-12
    )

    print("6. A3 moves humidity toward 82%: PASS")

    # --------------------------------------------------
    # 5. A4
    # --------------------------------------------------

    a4 = results["A4"][0]

    for feature in FEATURES:

        mask = changed_rows(
            original,
            a4,
            feature,
        )

        if feature in [
            "air_temperature",
            "humidity",
        ]:

            assert mask.sum() == 500

        else:

            assert mask.sum() == 0

    print(
        "7. A4 changes only temperature + humidity predictors: PASS"
    )

    # --------------------------------------------------
    # 6. A1
    # --------------------------------------------------

    a1 = results["A1"][0]

    for feature in ENV_FEATURES:

        mask = changed_rows(
            original,
            a1,
            feature,
        )

        assert mask.sum() == 500

    growth_stage_changed = changed_rows(
        original,
        a1,
        "growth_stage",
    )

    assert growth_stage_changed.sum() == 0

    print(
        "8. A1 jitters all 7 environmental features "
        "but not growth_stage: PASS"
    )

    # --------------------------------------------------
    # 7. LABEL AUDIT
    # --------------------------------------------------

    print()
    print("Label residual audit")
    print("-" * 72)

    for augmentation_id in [
        "A1",
        "A2",
        "A3",
        "A4",
    ]:

        augmented = results[augmentation_id][0]

        selected = augmented.iloc[
            reference_positions
        ]

        deterministic = deterministic_growth(
            selected
        )

        actual_target = selected[
            "target_growth"
        ].to_numpy()

        residual = (
            actual_target - deterministic
        )

        mean_residual = residual.mean()
        sd_residual = residual.std(ddof=1)

        print(
            f"{augmentation_id}: "
            f"noise mean={mean_residual:.4f}, "
            f"noise SD={sd_residual:.4f}"
        )

        # Broad sanity bounds, NOT inferential tests.
        assert abs(mean_residual) < 0.05

        assert 0.20 < sd_residual < 0.30

    print(
        "9. Oracle-relabel residual distribution: PASS"
    )

    # --------------------------------------------------
    # 8. IDENTIFIERS / METADATA
    # --------------------------------------------------

    for augmentation_id in [
        "A1",
        "A2",
        "A3",
        "A4",
    ]:

        augmented = results[augmentation_id][0]

        for column in [
            "domain",
            "sample_id",
            "dataset_id",
        ]:

            assert (
                original[column].equals(
                    augmented[column]
                )
            )

    print(
        "10. Domain/sample/dataset identifiers unchanged: PASS"
    )

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print()
    print("=" * 72)
    print("STRONG AUGMENTATION AUDIT PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()