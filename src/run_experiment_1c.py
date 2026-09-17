"""
Experiment 1C
=============

Research question:
How does training-data composition affect model
generalization as distribution-shift severity increases?

Independent variables:
    D = training-data composition
    S = shift type
    V = shift severity

Fixed:
    model = M0
    augmentation = A0
    training procedure = same as Experiment 1
    validation = normal.csv
    seeds = 11, 22, 33

IMPORTANT:
severity_test datasets are TEST ONLY.
"""

from pathlib import Path
import pandas as pd

from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = Path("data/training")
VALIDATION_DIR = Path("data/validation")
SEVERITY_DIR = Path("data/severity_test")

RESULT_DIR = Path("results/experiment_1c")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING COMPOSITIONS
# ============================================================

DATASETS = [
    "D00",
    "D01",
    "D02",
    "D03",
    "D04",
    "D05",
    "D06",
    "D07",
    "D08",
    "D09",
]


# ============================================================
# RANDOM SEEDS
# ============================================================

SEEDS = [
    11,
    22,
    33,
]


# ============================================================
# TEMPERATURE SEVERITY
# ============================================================

TEMPERATURE_LEVELS = [
    23.0,
    25.0,
    27.0,
    29.0,
    31.0,
    33.0,
]


# ============================================================
# HUMIDITY SEVERITY
# ============================================================

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
# BUILD TEST ENVIRONMENT TABLE
# ============================================================

def build_test_environments():

    environments = []

    # Temperature severity
    for severity, value in enumerate(
        TEMPERATURE_LEVELS
    ):

        environments.append({
            "shift_type": "TEMPERATURE",
            "severity": severity,
            "shift_value": value,
            "filename": f"temp_s{severity}.csv",
        })

    # Humidity severity
    for severity, value in enumerate(
        HUMIDITY_LEVELS
    ):

        environments.append({
            "shift_type": "HUMIDITY",
            "severity": severity,
            "shift_value": value,
            "filename": f"humidity_s{severity}.csv",
        })

    return environments


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("EXPERIMENT 1C")
    print("SHIFT-SEVERITY GENERALIZATION")
    print("=" * 70)

    environments = build_test_environments()

    print(
        f"Training compositions: {len(DATASETS)}"
    )

    print(
        f"Seeds: {len(SEEDS)}"
    )

    print(
        f"Severity environments: {len(environments)}"
    )

    total_models = (
        len(DATASETS)
        * len(SEEDS)
    )

    total_evaluations = (
        total_models
        * len(environments)
    )

    print(
        f"Models to train: {total_models}"
    )

    print(
        f"Evaluations expected: {total_evaluations}"
    )


    # ========================================================
    # VALIDATION DATA
    # ========================================================

    validation_df = pd.read_csv(
        VALIDATION_DIR / "normal.csv"
    )


    # ========================================================
    # PRELOAD TEST DATASETS
    # ========================================================

    severity_data = {}

    for environment in environments:

        filename = environment["filename"]

        path = (
            SEVERITY_DIR /
            filename
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing severity dataset: {path}"
            )

        severity_data[filename] = (
            pd.read_csv(path)
        )


    # ========================================================
    # RUN EXPERIMENT
    # ========================================================

    results = []

    model_number = 0


    for dataset_id in DATASETS:

        print()
        print("=" * 70)
        print(
            f"Training composition: {dataset_id}"
        )
        print("=" * 70)


        # ----------------------------------------------------
        # Load original training dataset
        # ----------------------------------------------------

        train_df = pd.read_csv(
            TRAIN_DIR /
            f"{dataset_id}.csv"
        )


        for seed in SEEDS:

            model_number += 1

            print()
            print(
                f"Training model "
                f"{model_number}/{total_models}"
            )

            print(
                f"Dataset={dataset_id}, "
                f"Seed={seed}"
            )


            # =================================================
            # TRAIN EXACTLY AS BEFORE
            # =================================================

            model, scaler, validation_loss = (
                train_model(
                    train_df=train_df,
                    validation_df=validation_df,
                    seed=seed,
                )
            )


            # =================================================
            # TEST ALL SEVERITY LEVELS
            # =================================================

            for environment in environments:

                filename = (
                    environment["filename"]
                )

                test_df = (
                    severity_data[filename]
                )

                metrics = evaluate_model(
                    model=model,
                    scaler=scaler,
                    dataframe=test_df,
                )


                # ---------------------------------------------
                # RECORD RESULT
                # ---------------------------------------------

                results.append({

                    "dataset_id":
                        dataset_id,

                    "model_id":
                        "M0",

                    "augmentation_id":
                        "A0",

                    "seed":
                        seed,

                    "shift_type":
                        environment["shift_type"],

                    "severity":
                        environment["severity"],

                    "shift_value":
                        environment["shift_value"],

                    "mse":
                        metrics["mse"],

                    "mae":
                        metrics["mae"],

                    "validation_mse":
                        validation_loss,
                })


                print(
                    f"{environment['shift_type']:11} "
                    f"S{environment['severity']} "
                    f"value="
                    f"{environment['shift_value']:5.1f} "
                    f"MSE="
                    f"{metrics['mse']:.4f}"
                )


    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULT_DIR /
        "runs_1c.csv"
    )

    results_df.to_csv(
        output_file,
        index=False,
    )


    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if len(results_df) != total_evaluations:

        raise ValueError(
            "Unexpected number of evaluation records. "
            f"Expected {total_evaluations}, "
            f"received {len(results_df)}."
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 1C COMPLETE")
    print("=" * 70)

    print(
        f"Models trained: "
        f"{total_models}"
    )

    print(
        f"Evaluation records: "
        f"{len(results_df)}"
    )

    print(
        f"Results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()