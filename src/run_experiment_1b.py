"""
Experiment 1B

Evaluate the Experiment 1 model/training setup on
three genuinely held-out OOD environments.

Training datasets:
    D00-D09

Model:
    M0

Augmentation:
    A0

Seeds:
    11, 22, 33

OOD environments:
    EXTREME_WARM
    EXTREME_HUMID
    WARM_HUMID
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

OOD_DIR = Path("data/ood_test")

RESULT_DIR = Path("results/experiment_1b")

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING DATA COMPOSITIONS
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
# HELD-OUT OOD ENVIRONMENTS
# ============================================================

OOD_ENVIRONMENTS = {

    "EXTREME_WARM":
        "extreme_warm.csv",

    "EXTREME_HUMID":
        "extreme_humid.csv",

    "WARM_HUMID":
        "warm_humid.csv",
}


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    results = []

    # --------------------------------------------------------
    # Same validation environment as Experiment 1
    # --------------------------------------------------------

    validation_df = pd.read_csv(
        VALIDATION_DIR / "normal.csv"
    )

    total_models = (
        len(DATASETS)
        * len(SEEDS)
    )

    model_number = 0

    print()
    print("=" * 70)
    print("EXPERIMENT 1B")
    print("HELD-OUT OOD GENERALIZATION")
    print("=" * 70)

    print(
        f"Training compositions: {len(DATASETS)}"
    )

    print(
        f"Seeds: {len(SEEDS)}"
    )

    print(
        f"Models to train: {total_models}"
    )

    print(
        f"OOD environments: {len(OOD_ENVIRONMENTS)}"
    )

    print()


    # ========================================================
    # LOOP THROUGH D00-D09
    # ========================================================

    for dataset_id in DATASETS:

        print()
        print("=" * 70)
        print(
            f"Training composition: {dataset_id}"
        )
        print("=" * 70)

        # Load original Experiment 1 training data
        train_df = pd.read_csv(
            TRAIN_DIR / f"{dataset_id}.csv"
        )


        # ====================================================
        # LOOP THROUGH SEEDS
        # ====================================================

        for seed in SEEDS:

            model_number += 1

            print()
            print(
                f"Training model "
                f"{model_number}/{total_models}"
            )

            print(
                f"Dataset={dataset_id}, Seed={seed}"
            )


            # ------------------------------------------------
            # Train EXACTLY as Experiment 1
            # ------------------------------------------------

            model, scaler, validation_loss = (
                train_model(
                    train_df=train_df,
                    validation_df=validation_df,
                    seed=seed,
                )
            )


            # =================================================
            # EVALUATE HELD-OUT OOD ENVIRONMENTS
            # =================================================

            for shift, filename in OOD_ENVIRONMENTS.items():

                test_df = pd.read_csv(
                    OOD_DIR / filename
                )

                metrics = evaluate_model(
                    model=model,
                    scaler=scaler,
                    dataframe=test_df,
                )


                # ---------------------------------------------
                # Store result
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

                    "shift":
                        shift,

                    "mse":
                        metrics["mse"],

                    "mae":
                        metrics["mae"],

                    "validation_mse":
                        validation_loss,
                })


                print(
                    f"{shift:16} "
                    f"MSE={metrics['mse']:.4f} "
                    f"MAE={metrics['mae']:.4f}"
                )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULT_DIR
        / "runs_1b.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )


    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("EXPERIMENT 1B COMPLETE")
    print("=" * 70)

    print(
        f"Models trained: {total_models}"
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