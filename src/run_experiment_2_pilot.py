from pathlib import Path
import json
import time

import pandas as pd

from src.augmentation import apply_augmentation
from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 2 ENGINEERING PILOT
# ============================================================
#
# Frozen pilot:
#
# D = D00, D05, D07
# A = A0, A1, A2, A3, A4
# seeds = 11, 22, 33
#
# 3 × 5 × 3 = 45 trained models
#
# Each model:
#   Experiment 1   = 4 environments
#   Experiment 1B  = 3 environments
#   Experiment 1C  = 13 environments
#
# Total:
#   45 × 20 = 900 evaluation records
#
# ENGINEERING PILOT ONLY.
# Do not use this pilot to tune augmentation definitions.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
OOD_DIR = DATA_DIR / "ood_test"
SEVERITY_DIR = DATA_DIR / "severity_test"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_2"
    / "pilot"
)

CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# FROZEN PILOT DESIGN
# ------------------------------------------------------------

DATASETS = [
    "D00",
    "D05",
    "D07",
]

AUGMENTATIONS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]

SEEDS = [
    11,
    22,
    33,
]


# ------------------------------------------------------------
# Load validation data
#
# Same checkpoint-selection environment as Phase 1.
# ------------------------------------------------------------

validation_df = pd.read_csv(
    VALIDATION_DIR / "normal.csv"
)


# ------------------------------------------------------------
# Evaluation environments
# ------------------------------------------------------------

evaluation_environments = []


# ============================================================
# EXPERIMENT 1
# ============================================================

for environment in [
    "normal",
    "warm",
    "humid",
    "cool",
]:

    evaluation_environments.append(
        {
            "experiment": "EXP1",
            "shift_type": environment,
            "severity": None,
            "environment": environment,
            "path": TEST_DIR / f"{environment}.csv",
        }
    )


# ============================================================
# EXPERIMENT 1B
# ============================================================

for environment in [
    "extreme_warm",
    "extreme_humid",
    "warm_humid",
]:

    evaluation_environments.append(
        {
            "experiment": "EXP1B",
            "shift_type": environment,
            "severity": None,
            "environment": environment,
            "path": OOD_DIR / f"{environment}.csv",
        }
    )


# ============================================================
# EXPERIMENT 1C
# Temperature severity
# ============================================================

TEMPERATURE_LEVELS = [
    23,
    25,
    27,
    29,
    31,
    33,
]

for severity, value in enumerate(
    TEMPERATURE_LEVELS
):

    evaluation_environments.append(
        {
            "experiment": "EXP1C",
            "shift_type": "temperature",
            "severity": severity,
            "environment": f"temp_s{severity}",
            "shift_value": value,
            "path": (
                SEVERITY_DIR
                / f"temp_s{severity}.csv"
            ),
        }
    )


# ============================================================
# EXPERIMENT 1C
# Humidity severity
# ============================================================

HUMIDITY_LEVELS = [
    65,
    70,
    75,
    80,
    85,
    90,
    92,
]

for severity, value in enumerate(
    HUMIDITY_LEVELS
):

    evaluation_environments.append(
        {
            "experiment": "EXP1C",
            "shift_type": "humidity",
            "severity": severity,
            "environment": f"humidity_s{severity}",
            "shift_value": value,
            "path": (
                SEVERITY_DIR
                / f"humidity_s{severity}.csv"
            ),
        }
    )


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

assert len(evaluation_environments) == 20


# ------------------------------------------------------------
# Preload evaluation data
# ------------------------------------------------------------

evaluation_data = {}

for environment_info in evaluation_environments:

    path = environment_info["path"]

    if not path.exists():

        raise FileNotFoundError(
            f"Missing evaluation dataset: {path}"
        )

    evaluation_data[
        environment_info["environment"]
    ] = pd.read_csv(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("PHASE 2 ENGINEERING PILOT")
    print("=" * 76)

    print()
    print("Datasets:", DATASETS)
    print("Augmentations:", AUGMENTATIONS)
    print("Seeds:", SEEDS)

    expected_models = (
        len(DATASETS)
        * len(AUGMENTATIONS)
        * len(SEEDS)
    )

    expected_records = (
        expected_models
        * len(evaluation_environments)
    )

    print()
    print(
        f"Expected models: {expected_models}"
    )

    print(
        f"Evaluation environments: "
        f"{len(evaluation_environments)}"
    )

    print(
        f"Expected evaluation records: "
        f"{expected_records}"
    )

    assert expected_models == 45
    assert expected_records == 900

    results = []

    model_counter = 0

    start_time = time.time()


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for dataset_id in DATASETS:

        training_path = (
            TRAINING_DIR
            / f"{dataset_id}.csv"
        )

        if not training_path.exists():

            raise FileNotFoundError(
                f"Missing training dataset: "
                f"{training_path}"
            )

        original_train_df = pd.read_csv(
            training_path
        )


        # ----------------------------------------------------
        # Training dataset integrity
        # ----------------------------------------------------

        if len(original_train_df) != 1000:

            raise ValueError(
                f"{dataset_id} has "
                f"{len(original_train_df)} rows. "
                f"Expected 1000."
            )


        for seed in SEEDS:

            # ------------------------------------------------
            # IMPORTANT:
            #
            # augmentation seed is separated from
            # model-training seed.
            # ------------------------------------------------

            augmentation_seed = (
                100000 + seed
            )


            for augmentation_id in AUGMENTATIONS:

                model_counter += 1

                model_id = (
                    f"{dataset_id}_"
                    f"{augmentation_id}_"
                    f"seed{seed}"
                )

                print()
                print("-" * 76)

                print(
                    f"[{model_counter}/"
                    f"{expected_models}] "
                    f"{model_id}"
                )

                # ============================================
                # AUGMENTATION
                # ============================================

                augmented_train_df, metadata = (
                    apply_augmentation(
                        original_train_df,
                        augmentation_id,
                        seed=augmentation_seed,
                        fraction=0.50,
                    )
                )


                # --------------------------------------------
                # Safety checks
                # --------------------------------------------

                assert (
                    len(augmented_train_df)
                    == 1000
                )

                assert not (
                    augmented_train_df
                    .isna()
                    .any()
                    .any()
                )

                if augmentation_id == "A0":

                    assert (
                        metadata["n_augmented"]
                        == 0
                    )

                else:

                    assert (
                        metadata["n_augmented"]
                        == 500
                    )


                # ============================================
                # TRAIN MODEL
                # ============================================

                model, scaler, best_validation_loss = (
                    train_model(
                        augmented_train_df,
                        validation_df,
                        seed=seed,
                    )
                )


                print(
                    "Best validation loss:",
                    f"{best_validation_loss:.6f}"
                )


                # ============================================
                # EVALUATION
                # ============================================

                for environment_info in (
                    evaluation_environments
                ):

                    environment_name = (
                        environment_info[
                            "environment"
                        ]
                    )

                    test_df = (
                        evaluation_data[
                            environment_name
                        ]
                    )

                    metrics = evaluate_model(
                        model,
                        scaler,
                        test_df,
                    )


                    record = {
                        "phase": "PHASE2",
                        "run_type": "PILOT",

                        "model_id": model_id,

                        "dataset_id": dataset_id,
                        "augmentation_id":
                            augmentation_id,

                        "model_config": "M0",

                        "seed": seed,
                        "augmentation_seed":
                            augmentation_seed,

                        "n_train":
                            len(
                                augmented_train_df
                            ),

                        "n_augmented":
                            metadata[
                                "n_augmented"
                            ],

                        "augmentation_fraction":
                            metadata[
                                "fraction"
                            ],

                        "experiment":
                            environment_info[
                                "experiment"
                            ],

                        "environment":
                            environment_name,

                        "shift_type":
                            environment_info[
                                "shift_type"
                            ],

                        "severity":
                            environment_info.get(
                                "severity"
                            ),

                        "shift_value":
                            environment_info.get(
                                "shift_value"
                            ),

                        "best_validation_loss":
                            best_validation_loss,

                        "mse":
                            metrics["mse"],

                        "mae":
                            metrics["mae"],
                    }

                    results.append(record)


                print(
                    "Evaluations completed:",
                    len(
                        evaluation_environments
                    )
                )


                # ============================================
                # SAVE INTERMEDIATE RESULTS
                # ============================================

                intermediate_df = pd.DataFrame(
                    results
                )

                intermediate_df.to_csv(
                    RESULTS_DIR
                    / "runs_phase2_pilot.csv",
                    index=False,
                )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("FINAL INTEGRITY CHECK")
    print("=" * 76)


    print(
        "Rows:",
        len(results_df)
    )

    assert len(results_df) == 900


    unique_models = (
        results_df["model_id"]
        .nunique()
    )

    print(
        "Unique models:",
        unique_models
    )

    assert unique_models == 45


    duplicate_count = (
        results_df
        .duplicated(
            subset=[
                "model_id",
                "experiment",
                "environment",
            ]
        )
        .sum()
    )

    print(
        "Duplicate evaluation records:",
        duplicate_count
    )

    assert duplicate_count == 0


    # --------------------------------------------------------
    # Experiment counts
    # --------------------------------------------------------

    experiment_counts = (
        results_df[
            "experiment"
        ]
        .value_counts()
        .sort_index()
    )

    print()
    print("Experiment counts:")
    print(experiment_counts)


    assert (
        experiment_counts["EXP1"]
        == 180
    )

    assert (
        experiment_counts["EXP1B"]
        == 135
    )

    assert (
        experiment_counts["EXP1C"]
        == 585
    )


    # --------------------------------------------------------
    # Every model must have exactly 20 evaluations
    # --------------------------------------------------------

    evaluations_per_model = (
        results_df
        .groupby("model_id")
        .size()
    )

    assert (
        evaluations_per_model
        == 20
    ).all()

    print()
    print(
        "20 evaluations per model: PASS"
    )


    # --------------------------------------------------------
    # Every D × A × seed combination exactly once
    # --------------------------------------------------------

    model_design = (
        results_df[
            [
                "dataset_id",
                "augmentation_id",
                "seed",
                "model_id",
            ]
        ]
        .drop_duplicates()
    )

    assert len(model_design) == 45

    print(
        "45 unique D x A x seed models: PASS"
    )


    # --------------------------------------------------------
    # Save final
    # --------------------------------------------------------

    output_path = (
        RESULTS_DIR
        / "runs_phase2_pilot.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )


    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "phase": "PHASE2",
        "run_type": "PILOT",

        "datasets": DATASETS,
        "augmentations": AUGMENTATIONS,
        "seeds": SEEDS,

        "augmentation_fraction": 0.50,

        "augmentation_seed_rule":
            "100000 + model training seed",

        "model_config": "M0",

        "expected_models":
            expected_models,

        "evaluation_environments":
            len(evaluation_environments),

        "expected_records":
            expected_records,

        "experiment_counts": {
            "EXP1": 180,
            "EXP1B": 135,
            "EXP1C": 585,
        },

        "purpose":
            "Engineering pilot only. "
            "Not for augmentation tuning.",
    }

    with open(
        RESULTS_DIR
        / "pilot_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=2,
        )


    elapsed = (
        time.time()
        - start_time
    )


    print()
    print(
        f"Saved results to:"
        f"\n{output_path}"
    )

    print(
        f"\nRuntime: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 76)
    print(
        "PHASE 2 ENGINEERING PILOT COMPLETED"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()