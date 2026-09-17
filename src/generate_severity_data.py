"""
Experiment 1C
Controlled distribution-shift severity datasets.

Temperature experiment:
    Change air temperature only.

Humidity experiment:
    Change humidity only.

Everything else remains at the normal-domain distribution.

These datasets are TEST ONLY.
"""

from pathlib import Path
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/severity_test")

N_SAMPLES = 6000


# ============================================================
# NORMAL BASELINE
# ============================================================

NORMAL_PARAMS = {
    "air_temperature": (23.0, 1.5),
    "humidity": (65.0, 5.0),
    "water_temperature": (20.0, 1.0),
    "ph": (6.0, 0.15),
    "ec": (1.7, 0.20),
    "light": (450.0, 60.0),
    "co2": (750.0, 80.0),
}


# ============================================================
# SEVERITY LEVELS
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
# GENERATE FEATURES
# ============================================================

def generate_domain(
    feature_to_shift,
    shifted_mean,
    n_samples,
    seed,
    domain_name,
):

    rng = np.random.default_rng(seed)

    data = {}

    for feature, (mean, std) in NORMAL_PARAMS.items():

        if feature == feature_to_shift:
            mean = shifted_mean

        data[feature] = rng.normal(
            loc=mean,
            scale=std,
            size=n_samples,
        )

    data["growth_stage"] = rng.uniform(
        0.0,
        1.0,
        size=n_samples,
    )

    df = pd.DataFrame(data)

    df["domain"] = domain_name

    df["sample_id"] = [
        f"{domain_name}_{seed}_{i}"
        for i in range(n_samples)
    ]

    return df


# ============================================================
# SAME TARGET FUNCTION AS EXPERIMENTS 1 AND 1B
# ============================================================

def add_target(df, seed):

    rng = np.random.default_rng(seed)

    t = df["air_temperature"].to_numpy()
    h = df["humidity"].to_numpy()
    tw = df["water_temperature"].to_numpy()
    ph = df["ph"].to_numpy()
    ec = df["ec"].to_numpy()
    light = df["light"].to_numpy()
    co2 = df["co2"].to_numpy()
    stage = df["growth_stage"].to_numpy()

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

    noise = rng.normal(
        0.0,
        0.25,
        size=len(df),
    )

    df = df.copy()

    df["target_growth"] = (
        growth + noise
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate(df, expected_rows):

    assert len(df) == expected_rows

    assert not df.isnull().any().any()

    assert not df["sample_id"].duplicated().any()

    assert "target_growth" in df.columns


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("EXPERIMENT 1C")
    print("GENERATING SHIFT-SEVERITY DATASETS")
    print("=" * 70)


    # ========================================================
    # TEMPERATURE SEVERITY
    # ========================================================

    print("\nTEMPERATURE SEVERITY")

    for severity, temperature in enumerate(
        TEMPERATURE_LEVELS
    ):

        name = f"temp_s{severity}"

        df = generate_domain(
            feature_to_shift="air_temperature",
            shifted_mean=temperature,
            n_samples=N_SAMPLES,
            seed=10000 + severity,
            domain_name=name,
        )

        df = add_target(
            df,
            seed=11000 + severity,
        )

        validate(
            df,
            N_SAMPLES,
        )

        output = (
            OUTPUT_DIR /
            f"{name}.csv"
        )

        df.to_csv(
            output,
            index=False,
        )

        print(
            f"S{severity}: "
            f"T={temperature:4.1f} "
            f"rows={len(df)} "
            f"-> {output}"
        )


    # ========================================================
    # HUMIDITY SEVERITY
    # ========================================================

    print("\nHUMIDITY SEVERITY")

    for severity, humidity in enumerate(
        HUMIDITY_LEVELS
    ):

        name = f"humidity_s{severity}"

        df = generate_domain(
            feature_to_shift="humidity",
            shifted_mean=humidity,
            n_samples=N_SAMPLES,
            seed=12000 + severity,
            domain_name=name,
        )

        df = add_target(
            df,
            seed=13000 + severity,
        )

        validate(
            df,
            N_SAMPLES,
        )

        output = (
            OUTPUT_DIR /
            f"{name}.csv"
        )

        df.to_csv(
            output,
            index=False,
        )

        print(
            f"S{severity}: "
            f"H={humidity:4.1f} "
            f"rows={len(df)} "
            f"-> {output}"
        )


    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Temperature datasets:",
        len(TEMPERATURE_LEVELS)
    )

    print(
        "Humidity datasets:",
        len(HUMIDITY_LEVELS)
    )

    print(
        "Total datasets:",
        len(TEMPERATURE_LEVELS)
        + len(HUMIDITY_LEVELS)
    )


if __name__ == "__main__":
    main()