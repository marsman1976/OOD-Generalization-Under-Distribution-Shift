from pathlib import Path

import pandas as pd

from src.train import train_model
from src.evaluate import evaluate_model


TRAIN_DIR = Path("data/training")
VALIDATION_DIR = Path("data/validation")
TEST_DIR = Path("data/test")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(
    exist_ok=True
)


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


SEEDS = [
    11,
    22,
    33
]


TEST_ENVIRONMENTS = {
    "ID": "normal.csv",
    "WARM": "warm.csv",
    "HUMID": "humid.csv",
    "COOL": "cool.csv",
}


def main():

    results = []

    # We use normal validation data
    # for the baseline experiment.

    validation_df = pd.read_csv(
        VALIDATION_DIR / "normal.csv"
    )

    total_models = (
        len(DATASETS)
        *
        len(SEEDS)
    )

    model_number = 0

    for dataset_id in DATASETS:

        print()
        print("=" * 60)
        print(f"Dataset: {dataset_id}")
        print("=" * 60)

        train_df = pd.read_csv(
            TRAIN_DIR /
            f"{dataset_id}.csv"
        )

        for seed in SEEDS:

            model_number += 1

            print(
                f"\nTraining model "
                f"{model_number}/{total_models}"
            )

            print(
                f"Dataset={dataset_id}, "
                f"Seed={seed}"
            )

            model, scaler, val_loss = train_model(
                train_df=train_df,
                validation_df=validation_df,
                seed=seed
            )

            # ------------------------
            # Evaluate all environments
            # ------------------------

            for shift, filename in (
                TEST_ENVIRONMENTS.items()
            ):

                test_df = pd.read_csv(
                    TEST_DIR / filename
                )

                metrics = evaluate_model(
                    model,
                    scaler,
                    test_df
                )

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
                        val_loss,
                })

                print(
                    f"{shift:6} "
                    f"MSE={metrics['mse']:.4f} "
                    f"MAE={metrics['mae']:.4f}"
                )

    # ------------------------
    # Save everything
    # ------------------------

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        RESULT_DIR /
        "runs.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

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