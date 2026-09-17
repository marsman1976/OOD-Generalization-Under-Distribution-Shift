from pathlib import Path
import json
import time

import pandas as pd

from src.augmentation import apply_augmentation
from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 2 CONFIRMATION
#
# D00-D09 × A0-A4 × 10 seeds = 500 trained models
# 20 evaluations/model = 10,000 evaluation records
#
# Frozen after Phase 2 engineering pilot.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
TRAIN_DIR = DATA / "training"
VALIDATION_DIR = DATA / "validation"
TEST_DIR = DATA / "test"
OOD_DIR = DATA / "ood_test"
SEVERITY_DIR = DATA / "severity_test"

OUTPUT = (
    ROOT / "results" /
    "experiment_2" /
    "confirmation"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT / "master_results.csv"
CONFIG_FILE = OUTPUT / "confirmation_config.json"


DATASETS = [
    f"D{i:02d}"
    for i in range(10)
]

AUGMENTATIONS = [
    "A0", "A1", "A2", "A3", "A4"
]

SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]

AUGMENTATION_FRACTION = 0.50


# ============================================================
# Evaluation environments
# ============================================================

ENVIRONMENTS = []


# EXP1

for env in [
    "normal",
    "warm",
    "humid",
    "cool",
]:
    ENVIRONMENTS.append({
        "experiment": "EXP1",
        "environment": env,
        "shift_type": env,
        "severity": None,
        "shift_value": None,
        "path": TEST_DIR / f"{env}.csv",
    })


# EXP1B

for env in [
    "extreme_warm",
    "extreme_humid",
    "warm_humid",
]:
    ENVIRONMENTS.append({
        "experiment": "EXP1B",
        "environment": env,
        "shift_type": env,
        "severity": None,
        "shift_value": None,
        "path": OOD_DIR / f"{env}.csv",
    })


# EXP1C temperature

TEMPERATURE_LEVELS = [
    23, 25, 27, 29, 31, 33
]

for severity, value in enumerate(
    TEMPERATURE_LEVELS
):
    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": f"temp_s{severity}",
        "shift_type": "temperature",
        "severity": severity,
        "shift_value": value,
        "path": (
            SEVERITY_DIR /
            f"temp_s{severity}.csv"
        ),
    })


# EXP1C humidity

HUMIDITY_LEVELS = [
    65, 70, 75, 80, 85, 90, 92
]

for severity, value in enumerate(
    HUMIDITY_LEVELS
):
    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": f"humidity_s{severity}",
        "shift_type": "humidity",
        "severity": severity,
        "shift_value": value,
        "path": (
            SEVERITY_DIR /
            f"humidity_s{severity}.csv"
        ),
    })


assert len(ENVIRONMENTS) == 20


# ============================================================
# Preload evaluation datasets
# ============================================================

evaluation_data = {}

for info in ENVIRONMENTS:

    if not info["path"].exists():
        raise FileNotFoundError(
            info["path"]
        )

    evaluation_data[
        info["environment"]
    ] = pd.read_csv(
        info["path"]
    )


validation_df = pd.read_csv(
    VALIDATION_DIR / "normal.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "phase": "PHASE2",
    "run_type": "CONFIRMATION",
    "datasets": DATASETS,
    "augmentations": AUGMENTATIONS,
    "seeds": SEEDS,
    "augmentation_fraction":
        AUGMENTATION_FRACTION,
    "augmentation_seed_rule":
        "100000 + training seed",
    "model_config": "M0",
    "n_models": 500,
    "n_environments": 20,
    "expected_records": 10000,
    "purpose":
        "Confirmatory Phase 2 D x A x S x V experiment",
}


# ============================================================
# Existing completed models
# ============================================================

def load_existing():

    if not RESULTS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        RESULTS_FILE
    )

    print(
        f"Existing records found: {len(df)}"
    )

    return df


def completed_model_ids(df):

    if df.empty:
        return set()

    counts = (
        df.groupby("model_id")
        .size()
    )

    # A model is considered complete ONLY
    # if all 20 evaluations exist.
    return set(
        counts[
            counts == 20
        ].index
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 76)
    print("PHASE 2 CONFIRMATION")
    print("=" * 76)

    expected_models = (
        len(DATASETS)
        * len(AUGMENTATIONS)
        * len(SEEDS)
    )

    expected_records = (
        expected_models
        * len(ENVIRONMENTS)
    )

    assert expected_models == 500
    assert expected_records == 10000

    print("Datasets:", len(DATASETS))
    print("Augmentations:", len(AUGMENTATIONS))
    print("Seeds:", len(SEEDS))
    print("Models:", expected_models)
    print("Environments/model:", 20)
    print("Expected records:", expected_records)

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            CONFIG,
            f,
            indent=2,
        )

    existing = load_existing()

    completed = completed_model_ids(
        existing
    )

    print(
        "Completed models already present:",
        len(completed)
    )

    new_records = []

    trained_now = 0

    start = time.time()


    # ========================================================
    # Training
    # ========================================================

    for dataset_id in DATASETS:

        train_path = (
            TRAIN_DIR /
            f"{dataset_id}.csv"
        )

        if not train_path.exists():
            raise FileNotFoundError(
                train_path
            )

        original_train = pd.read_csv(
            train_path
        )

        assert len(original_train) == 1000


        for seed in SEEDS:

            augmentation_seed = (
                100000 + seed
            )


            for augmentation_id in AUGMENTATIONS:

                model_id = (
                    f"{dataset_id}_"
                    f"{augmentation_id}_"
                    f"seed{seed}"
                )


                # --------------------------------------------
                # Resume support
                # --------------------------------------------

                if model_id in completed:

                    print(
                        f"SKIP complete: {model_id}"
                    )

                    continue


                print()
                print("-" * 76)
                print(
                    "Training:",
                    model_id
                )


                # ============================================
                # Augmentation
                # ============================================

                train_df, metadata = (
                    apply_augmentation(
                        original_train,
                        augmentation_id,
                        seed=augmentation_seed,
                        fraction=
                            AUGMENTATION_FRACTION,
                    )
                )

                assert len(train_df) == 1000
                assert not train_df.isna().any().any()

                if augmentation_id == "A0":
                    assert metadata[
                        "n_augmented"
                    ] == 0
                else:
                    assert metadata[
                        "n_augmented"
                    ] == 500


                # ============================================
                # Training
                # ============================================

                model, scaler, val_loss = (
                    train_model(
                        train_df,
                        validation_df,
                        seed=seed,
                    )
                )


                # ============================================
                # Evaluate same fitted model 20 times
                # ============================================

                model_records = []

                for info in ENVIRONMENTS:

                    test_df = (
                        evaluation_data[
                            info["environment"]
                        ]
                    )

                    metrics = evaluate_model(
                        model,
                        scaler,
                        test_df,
                    )

                    model_records.append({
                        "phase": "PHASE2",
                        "run_type":
                            "CONFIRMATION",

                        "model_id":
                            model_id,

                        "dataset_id":
                            dataset_id,

                        "augmentation_id":
                            augmentation_id,

                        "model_config": "M0",

                        "seed": seed,

                        "augmentation_seed":
                            augmentation_seed,

                        "n_train": 1000,

                        "n_augmented":
                            metadata[
                                "n_augmented"
                            ],

                        "augmentation_fraction":
                            metadata[
                                "fraction"
                            ],

                        "experiment":
                            info["experiment"],

                        "environment":
                            info["environment"],

                        "shift_type":
                            info["shift_type"],

                        "severity":
                            info["severity"],

                        "shift_value":
                            info["shift_value"],

                        "best_validation_loss":
                            val_loss,

                        "mse":
                            metrics["mse"],

                        "mae":
                            metrics["mae"],
                    })


                assert len(model_records) == 20

                new_records.extend(
                    model_records
                )

                trained_now += 1


                # ============================================
                # SAVE AFTER EVERY MODEL
                # ============================================

                new_df = pd.DataFrame(
                    new_records
                )

                if existing.empty:
                    combined = new_df
                else:
                    combined = pd.concat(
                        [
                            existing,
                            new_df,
                        ],
                        ignore_index=True,
                    )

                # Remove incomplete/duplicate remnants
                # for this newly completed model.
                combined = (
                    combined
                    .drop_duplicates(
                        subset=[
                            "model_id",
                            "experiment",
                            "environment",
                        ],
                        keep="last",
                    )
                )

                combined.to_csv(
                    RESULTS_FILE,
                    index=False,
                )

                print(
                    f"Completed this session: "
                    f"{trained_now}"
                )

                print(
                    f"Total records saved: "
                    f"{len(combined)}"
                )


    # ========================================================
    # FINAL INTEGRITY
    # ========================================================

    final = pd.read_csv(
        RESULTS_FILE
    )

    print()
    print("=" * 76)
    print("FINAL INTEGRITY CHECK")
    print("=" * 76)

    print(
        "Records:",
        len(final)
    )

    print(
        "Unique models:",
        final["model_id"].nunique()
    )

    duplicates = (
        final.duplicated(
            subset=[
                "model_id",
                "experiment",
                "environment",
            ]
        ).sum()
    )

    print(
        "Duplicate records:",
        duplicates
    )

    counts = (
        final.groupby("model_id")
        .size()
    )

    incomplete = counts[
        counts != 20
    ]

    print(
        "Incomplete models:",
        len(incomplete)
    )

    experiment_counts = (
        final["experiment"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Experiment counts:")
    print(experiment_counts)


    # Expected:
    #
    # EXP1:
    # 500 × 4 = 2000
    #
    # EXP1B:
    # 500 × 3 = 1500
    #
    # EXP1C:
    # 500 × 13 = 6500

    assert len(final) == 10000

    assert (
        final["model_id"]
        .nunique()
        == 500
    )

    assert duplicates == 0

    assert len(incomplete) == 0

    assert (
        experiment_counts["EXP1"]
        == 2000
    )

    assert (
        experiment_counts["EXP1B"]
        == 1500
    )

    assert (
        experiment_counts["EXP1C"]
        == 6500
    )


    elapsed = (
        time.time() - start
    )

    print()
    print(
        f"Runtime this session: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 76)
    print(
        "PHASE 2 CONFIRMATION COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()