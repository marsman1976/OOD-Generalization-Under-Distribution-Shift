"""
Phase 6 augmentation integrity audit.

Purpose
-------
Validate the Phase 6 augmentation implementation before any
machine-learning training is performed.

This script checks:
- A0 leaves data unchanged
- A1-A4 preserve dataset size
- A1-A4 transform exactly 50% of observations
- reproducibility
- different-seed behavior
- directed temperature/humidity shifts
- G2 dependency propagation
- numerical validity
- frozen feature bounds

IMPORTANT:
This is an engineering/integrity audit only.
No ML model is trained.
No ML performance is inspected.
"""

import numpy as np
import pandas as pd

from src.generator_v2 import TRAIN_DIR

from src.augmentation_phase6 import (
    apply_phase6_augmentation,
)


# ============================================================
# Global audit counter
# ============================================================

checks_passed = 0


def check(condition, message):
    """
    Record a successful check or stop immediately if it fails.
    """

    global checks_passed

    if not condition:
        raise AssertionError(
            f"[FAIL] {message}"
        )

    checks_passed += 1
    print(f"[PASS] {message}")


def section(title):
    """
    Print formatted section header.
    """

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Load frozen Phase 6 D00
# ============================================================

section("PHASE 6 AUGMENTATION AUDIT")


path = TRAIN_DIR / "D00.csv"


check(
    path.exists(),
    "Frozen Phase 6 D00 exists",
)


original = pd.read_csv(path)


check(
    len(original) == 1000,
    "D00 contains 1000 observations",
)


required_columns = [
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


check(
    all(
        column in original.columns
        for column in required_columns
    ),
    "D00 contains all required Phase 6 columns",
)


# ============================================================
# 1. A0 — no augmentation
# ============================================================

section("1. A0 — NO AUGMENTATION")


a0, meta0 = apply_phase6_augmentation(
    df=original,
    augmentation_id="A0",
    seed=100251,
    fraction=0.50,
)


check(
    a0.equals(
        original.reset_index(drop=True)
    ),
    "A0 leaves data exactly unchanged",
)


check(
    meta0["n_augmented"] == 0,
    "A0 reports zero augmented observations",
)


check(
    len(a0) == 1000,
    "A0 preserves dataset size",
)


check(
    meta0["n_final"] == 1000,
    "A0 metadata reports n_final=1000",
)


# ============================================================
# 2. A1-A4 basic integrity
# ============================================================

section("2. A1-A4 BASIC INTEGRITY")


results = {}
metadata_results = {}


for augmentation_id in [
    "A1",
    "A2",
    "A3",
    "A4",
]:

    augmented, metadata = apply_phase6_augmentation(
        df=original,
        augmentation_id=augmentation_id,
        seed=100251,
        fraction=0.50,
    )

    results[augmentation_id] = augmented
    metadata_results[augmentation_id] = metadata

    check(
        len(augmented) == 1000,
        f"{augmentation_id} preserves n=1000",
    )

    check(
        metadata["n_augmented"] == 500,
        (
            f"{augmentation_id} transforms "
            "exactly 500 rows"
        ),
    )

    check(
        metadata["n_final"] == 1000,
        (
            f"{augmentation_id} metadata "
            "reports n_final=1000"
        ),
    )

    check(
        metadata["augmentation_fraction"] == 0.50,
        (
            f"{augmentation_id} metadata "
            "reports fraction=0.50"
        ),
    )

    numeric = augmented[required_columns]

    check(
        numeric.notna().all().all(),
        (
            f"{augmentation_id} contains "
            "no missing numeric values"
        ),
    )

    check(
        np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all(),
        (
            f"{augmentation_id} contains "
            "only finite numeric values"
        ),
    )


# ============================================================
# 3. Reproducibility
# ============================================================

section("3. REPRODUCIBILITY")


for augmentation_id in [
    "A1",
    "A2",
    "A3",
    "A4",
]:

    x, _ = apply_phase6_augmentation(
        df=original,
        augmentation_id=augmentation_id,
        seed=100251,
        fraction=0.50,
    )

    y, _ = apply_phase6_augmentation(
        df=original,
        augmentation_id=augmentation_id,
        seed=100251,
        fraction=0.50,
    )

    check(
        x.equals(y),
        (
            f"{augmentation_id}: same seed "
            "produces identical output"
        ),
    )


# ============================================================
# 4. Different seeds
# ============================================================

section("4. DIFFERENT-SEED CHECK")


for augmentation_id in [
    "A1",
    "A2",
    "A3",
    "A4",
]:

    x, _ = apply_phase6_augmentation(
        df=original,
        augmentation_id=augmentation_id,
        seed=100251,
        fraction=0.50,
    )

    y, _ = apply_phase6_augmentation(
        df=original,
        augmentation_id=augmentation_id,
        seed=100262,
        fraction=0.50,
    )

    check(
        not x.equals(y),
        (
            f"{augmentation_id}: different seeds "
            "produce different output"
        ),
    )


# ============================================================
# 5. Directed augmentation checks
# ============================================================

section("5. DIRECTED AUGMENTATION CHECKS")


original_temp = original[
    "air_temperature"
].mean()

original_humidity = original[
    "humidity"
].mean()


a1 = results["A1"]
a2 = results["A2"]
a3 = results["A3"]
a4 = results["A4"]


print()
print(
    f"Original temperature mean: "
    f"{original_temp:.4f}"
)
print(
    f"A1 temperature mean:       "
    f"{a1['air_temperature'].mean():.4f}"
)
print(
    f"A2 temperature mean:       "
    f"{a2['air_temperature'].mean():.4f}"
)
print(
    f"A3 temperature mean:       "
    f"{a3['air_temperature'].mean():.4f}"
)
print(
    f"A4 temperature mean:       "
    f"{a4['air_temperature'].mean():.4f}"
)

print()

print(
    f"Original humidity mean:    "
    f"{original_humidity:.4f}"
)
print(
    f"A1 humidity mean:          "
    f"{a1['humidity'].mean():.4f}"
)
print(
    f"A2 humidity mean:          "
    f"{a2['humidity'].mean():.4f}"
)
print(
    f"A3 humidity mean:          "
    f"{a3['humidity'].mean():.4f}"
)
print(
    f"A4 humidity mean:          "
    f"{a4['humidity'].mean():.4f}"
)


check(
    a2["air_temperature"].mean()
    > original_temp,
    "A2 shifts temperature upward",
)


check(
    a3["humidity"].mean()
    > original_humidity,
    "A3 shifts humidity upward",
)


check(
    a4["air_temperature"].mean()
    > original_temp,
    "A4 shifts temperature upward",
)


check(
    a4["humidity"].mean()
    > original_humidity,
    "A4 shifts humidity upward",
)


# ============================================================
# 6. G2 dependency propagation
# ============================================================

section("6. G2 DEPENDENCY PROPAGATION")


original_water = original[
    "water_temperature"
].mean()


a2_water = a2[
    "water_temperature"
].mean()

a4_water = a4[
    "water_temperature"
].mean()


print()
print(
    f"Original water-temperature mean: "
    f"{original_water:.4f}"
)
print(
    f"A2 water-temperature mean:       "
    f"{a2_water:.4f}"
)
print(
    f"A4 water-temperature mean:       "
    f"{a4_water:.4f}"
)


check(
    a2_water > original_water,
    (
        "A2 temperature shift propagates "
        "to water temperature"
    ),
)


check(
    a4_water > original_water,
    (
        "A4 temperature shift propagates "
        "to water temperature"
    ),
)


# A2 temperature increase should also produce the intended
# negative within-environment humidity response before the
# explicit humidity-directed augmentation used by A4.

a2_humidity = a2[
    "humidity"
].mean()


print(
    f"Original humidity mean:          "
    f"{original_humidity:.4f}"
)
print(
    f"A2 humidity mean:                "
    f"{a2_humidity:.4f}"
)


check(
    a2_humidity < original_humidity,
    (
        "A2 temperature shift propagates "
        "negative change to humidity"
    ),
)


# ============================================================
# 7. Frozen bounds
# ============================================================

section("7. FROZEN FEATURE BOUNDS")


for augmentation_id, df in results.items():

    check(
        df["humidity"].between(
            35.0,
            98.0,
        ).all(),
        (
            f"{augmentation_id}: "
            "humidity bounds preserved"
        ),
    )

    check(
        df["ph"].between(
            5.3,
            6.9,
        ).all(),
        (
            f"{augmentation_id}: "
            "pH bounds preserved"
        ),
    )

    check(
        df["ec"].between(
            0.8,
            2.8,
        ).all(),
        (
            f"{augmentation_id}: "
            "EC bounds preserved"
        ),
    )

    check(
        df["light"].between(
            100.0,
            850.0,
        ).all(),
        (
            f"{augmentation_id}: "
            "light bounds preserved"
        ),
    )

    check(
        df["co2"].between(
            350.0,
            1300.0,
        ).all(),
        (
            f"{augmentation_id}: "
            "CO2 bounds preserved"
        ),
    )

    check(
        df["growth_stage"].between(
            0.0,
            1.0,
        ).all(),
        (
            f"{augmentation_id}: "
            "growth-stage bounds preserved"
        ),
    )


# ============================================================
# 8. A1 local jitter
# ============================================================

section("8. A1 LOCAL JITTER")


check(
    not a1.equals(original),
    "A1 changes the dataset",
)


check(
    not np.isclose(
        a1["air_temperature"].std(),
        original["air_temperature"].std(),
    ),
    "A1 changes temperature distribution",
)


check(
    not np.isclose(
        a1["humidity"].std(),
        original["humidity"].std(),
    ),
    "A1 changes humidity distribution",
)


check(
    not np.isclose(
        a1["light"].std(),
        original["light"].std(),
    ),
    "A1 changes light distribution",
)


# ============================================================
# 9. Target relabeling
# ============================================================

section("9. TARGET RELABELING")


for augmentation_id in [
    "A1",
    "A2",
    "A3",
    "A4",
]:

    df = results[augmentation_id]

    check(
        np.isfinite(
            df["growth"].to_numpy(
                dtype=float
            )
        ).all(),
        (
            f"{augmentation_id}: "
            "recomputed growth labels are finite"
        ),
    )

    check(
        df["growth"].nunique() > 100,
        (
            f"{augmentation_id}: "
            "growth target retains substantial variation"
        ),
    )


# ============================================================
# 10. Column integrity
# ============================================================

section("10. COLUMN INTEGRITY")


for augmentation_id in [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]:

    if augmentation_id == "A0":
        df = a0
    else:
        df = results[augmentation_id]

    check(
        list(df.columns)
        == list(original.columns),
        (
            f"{augmentation_id}: "
            "column structure preserved"
        ),
    )


# ============================================================
# Final result
# ============================================================

section("AUDIT COMPLETE")


print(
    f"Total checks passed: "
    f"{checks_passed}"
)

print()

print(
    "PHASE 6 AUGMENTATION AUDIT: PASS"
)

print()

print(
    "Validated:"
)

print(
    "  - A0 unchanged"
)

print(
    "  - fixed-budget augmentation"
)

print(
    "  - deterministic reproducibility"
)

print(
    "  - directed shifts"
)

print(
    "  - G2 dependency propagation"
)

print(
    "  - feature bounds"
)

print(
    "  - target integrity"
)

print(
    "  - column integrity"
)

print()

print(
    "No ML model was trained."
)

print(
    "No ML performance was inspected."
)

print(
    "Do not modify Phase 6 augmentation "
    "based on later model performance."
)