import numpy as np
import pandas as pd

from src.augmentation import (
    apply_augmentation,
    AUGMENTATION_IDS,
)


DATASET = "data/training/D00.csv"

SEED = 11


def main():

    print("=" * 70)
    print("PHASE 2 AUGMENTATION VALIDATION")
    print("=" * 70)

    original = pd.read_csv(DATASET)

    print()
    print("Original dataset")
    print("Rows:", len(original))
    print("Columns:", list(original.columns))

    assert len(original) == 1000

    # -----------------------------------------------------
    # Test every augmentation
    # -----------------------------------------------------

    for augmentation_id in AUGMENTATION_IDS:

        print()
        print("=" * 70)
        print("Testing:", augmentation_id)
        print("=" * 70)

        augmented, metadata = apply_augmentation(
            original,
            augmentation_id,
            seed=SEED,
        )

        print("Rows:", len(augmented))
        print(
            "Augmented rows:",
            metadata["n_augmented"],
        )

        # ---------------------------------------------
        # Test 1
        # Same number of rows
        # ---------------------------------------------

        assert len(augmented) == len(original)

        # ---------------------------------------------
        # Test 2
        # No NaN
        # ---------------------------------------------

        assert not augmented.isna().any().any()

        # ---------------------------------------------
        # Test 3
        # A0 must be identical
        # ---------------------------------------------

        if augmentation_id == "A0":

            pd.testing.assert_frame_equal(
                original,
                augmented,
            )

            print("A0 exact identity: PASS")

        else:

            assert metadata["n_augmented"] == 500

            print(
                "Fixed 50% augmentation: PASS"
            )

        # ---------------------------------------------
        # Test 4
        # Reproducibility
        # ---------------------------------------------

        augmented_again, metadata_again = (
            apply_augmentation(
                original,
                augmentation_id,
                seed=SEED,
            )
        )

        pd.testing.assert_frame_equal(
            augmented,
            augmented_again,
        )

        assert (
            metadata["selected_positions"]
            ==
            metadata_again["selected_positions"]
        )

        print("Reproducibility: PASS")

        # ---------------------------------------------
        # Basic distribution report
        # ---------------------------------------------

        print()
        print("Means:")

        for feature in [
            "air_temperature",
            "humidity",
            "water_temperature",
            "ph",
            "ec",
            "light",
            "co2",
            "target_growth",
        ]:

            before = original[feature].mean()
            after = augmented[feature].mean()

            print(
                f"{feature:20s} "
                f"{before:10.4f} -> "
                f"{after:10.4f}"
            )

    print()
    print("=" * 70)
    print("ALL BASIC TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()