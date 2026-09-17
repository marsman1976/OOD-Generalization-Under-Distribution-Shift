from pathlib import Path
import copy
import json
import pandas as pd
import torch

from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = Path("data/training")
VALIDATION_DIR = Path("data/validation")
TEST_DIR = Path("data/test")
OOD_DIR = Path("data/ood_test")
SEVERITY_DIR = Path("data/severity_test")

RESULT_DIR = Path("results/phase1_confirmation")
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = [
    "D00", "D01", "D02", "D03", "D04",
    "D05", "D06", "D07", "D08", "D09",
]

# Confirmation seeds.
# The original pilot seeds are retained and seven new seeds are added.
SEEDS = [
    11, 22, 33, 44, 55,
    66, 77, 88, 99, 111,
]


# ============================================================
# EXPERIMENT 1 TEST ENVIRONMENTS
# ============================================================

EXPERIMENT_1 = {
    "ID": "normal.csv",
    "WARM": "warm.csv",
    "HUMID": "humid.csv",
    "COOL": "cool.csv",
}


# ============================================================
# EXPERIMENT 1B TEST ENVIRONMENTS
# ============================================================

EXPERIMENT_1B = {
    "EXTREME_WARM": "extreme_warm.csv",
    "EXTREME_HUMID": "extreme_humid.csv",
    "WARM_HUMID": "warm_humid.csv",
}


# ============================================================
# EXPERIMENT 1C SEVERITY LEVELS
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
# CHECK FILES
# ============================================================

def check_required_files():

    print("Checking required files...")

    for dataset_id in DATASETS:

        path = TRAIN_DIR / f"{dataset_id}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing training dataset: {path}"
            )

    validation_file = VALIDATION_DIR / "normal.csv"

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Missing validation file: {validation_file}"
        )

    for filename in EXPERIMENT_1.values():

        path = TEST_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Experiment 1 test file: {path}"
            )

    for filename in EXPERIMENT_1B.values():

        path = OOD_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Experiment 1B file: {path}"
            )

    for severity in range(len(TEMPERATURE_LEVELS)):

        path = SEVERITY_DIR / f"temp_s{severity}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing severity file: {path}"
            )

    for severity in range(len(HUMIDITY_LEVELS)):

        path = SEVERITY_DIR / f"humidity_s{severity}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing severity file: {path}"
            )

    print("All required files found.")


# ============================================================
# LOAD TEST DATA ONCE
# ============================================================

def load_evaluation_data():

    evaluation_data = {}

    # ---------------- Experiment 1 ----------------

    for shift, filename in EXPERIMENT_1.items():

        key = (
            "EXP1",
            shift,
            None,
            None,
        )

        evaluation_data[key] = pd.read_csv(
            TEST_DIR / filename
        )

    # ---------------- Experiment 1B ----------------

    for shift, filename in EXPERIMENT_1B.items():

        key = (
            "EXP1B",
            shift,
            None,
            None,
        )

        evaluation_data[key] = pd.read_csv(
            OOD_DIR / filename
        )

    # ---------------- Temperature severity ----------------

    for severity, value in enumerate(
        TEMPERATURE_LEVELS
    ):

        key = (
            "EXP1C",
            "TEMPERATURE",
            severity,
            value,
        )

        evaluation_data[key] = pd.read_csv(
            SEVERITY_DIR /
            f"temp_s{severity}.csv"
        )

    # ---------------- Humidity severity ----------------

    for severity, value in enumerate(
        HUMIDITY_LEVELS
    ):

        key = (
            "EXP1C",
            "HUMIDITY",
            severity,
            value,
        )

        evaluation_data[key] = pd.read_csv(
            SEVERITY_DIR /
            f"humidity_s{severity}.csv"
        )

    return evaluation_data


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    scaler,
    dataset_id,
    seed,
    validation_mse,
):

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"{dataset_id}_seed{seed}.pt"
    )

    checkpoint = {
        "dataset_id": dataset_id,
        "seed": seed,
        "model_id": "M0",
        "augmentation_id": "A0",
        "model_state_dict":
            copy.deepcopy(model.state_dict()),
        "scaler_mean":
            scaler.mean_,
        "scaler_scale":
            scaler.scale_,
        "scaler_var":
            scaler.var_,
        "validation_mse":
            validation_mse,
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    return checkpoint_path


# ============================================================
# EVALUATE ONE TRAINED MODEL
# ============================================================

def evaluate_all_environments(
    model,
    scaler,
    dataset_id,
    seed,
    validation_mse,
    evaluation_data,
):

    records = []

    for key, dataframe in evaluation_data.items():

        experiment = key[0]
        shift_type = key[1]
        severity = key[2]
        shift_value = key[3]

        metrics = evaluate_model(
            model,
            scaler,
            dataframe,
        )

        records.append({
            "dataset_id":
                dataset_id,

            "model_id":
                "M0",

            "augmentation_id":
                "A0",

            "seed":
                seed,

            "experiment":
                experiment,

            "shift_type":
                shift_type,

            "severity":
                severity,

            "shift_value":
                shift_value,

            "mse":
                metrics["mse"],

            "mae":
                metrics["mae"],

            "validation_mse":
                validation_mse,
        })

    return records


# ============================================================
# VALIDATE RESULTS
# ============================================================

def validate_results(results_df):

    expected_models = (
        len(DATASETS)
        * len(SEEDS)
    )

    expected_environments = (
        len(EXPERIMENT_1)
        + len(EXPERIMENT_1B)
        + len(TEMPERATURE_LEVELS)
        + len(HUMIDITY_LEVELS)
    )

    expected_rows = (
        expected_models
        * expected_environments
    )

    print()
    print("Validation")
    print("-" * 60)

    print(
        "Expected trained models:",
        expected_models
    )

    print(
        "Evaluation environments/model:",
        expected_environments
    )

    print(
        "Expected result rows:",
        expected_rows
    )

    print(
        "Actual result rows:",
        len(results_df)
    )

    if len(results_df) != expected_rows:

        raise ValueError(
            "Unexpected number of result rows."
        )

    duplicate_columns = [
        "dataset_id",
        "seed",
        "experiment",
        "shift_type",
        "severity",
        "shift_value",
    ]

    duplicates = results_df.duplicated(
        subset=duplicate_columns
    ).sum()

    if duplicates != 0:

        raise ValueError(
            f"Found {duplicates} duplicate records."
        )

    print("Duplicate records: 0")
    print("Result validation passed.")


# ============================================================
# SAVE EXPERIMENT-SPECIFIC TABLES
# ============================================================

def save_separate_results(master_df):

    exp1 = master_df[
        master_df["experiment"] == "EXP1"
    ].copy()

    exp1b = master_df[
        master_df["experiment"] == "EXP1B"
    ].copy()

    exp1c = master_df[
        master_df["experiment"] == "EXP1C"
    ].copy()

    exp1.to_csv(
        RESULT_DIR / "runs_exp1_confirmation.csv",
        index=False,
    )

    exp1b.to_csv(
        RESULT_DIR / "runs_exp1b_confirmation.csv",
        index=False,
    )

    exp1c.to_csv(
        RESULT_DIR / "runs_exp1c_confirmation.csv",
        index=False,
    )

    print()
    print("Experiment-specific rows:")
    print("EXP1 :", len(exp1))
    print("EXP1B:", len(exp1b))
    print("EXP1C:", len(exp1c))


# ============================================================
# SAVE EXPERIMENT CONFIG
# ============================================================

def save_config():

    config = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "model_id": "M0",
        "augmentation_id": "A0",
        "experiment_1":
            EXPERIMENT_1,
        "experiment_1b":
            EXPERIMENT_1B,
        "temperature_levels":
            TEMPERATURE_LEVELS,
        "humidity_levels":
            HUMIDITY_LEVELS,
        "number_of_datasets":
            len(DATASETS),
        "number_of_seeds":
            len(SEEDS),
        "number_of_models":
            len(DATASETS) * len(SEEDS),
    }

    with open(
        RESULT_DIR / "confirmation_config.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PHASE 1 CONFIRMATION EXPERIMENT")
    print("=" * 70)

    check_required_files()

    print()
    print("Loading evaluation datasets...")

    evaluation_data = load_evaluation_data()

    print(
        "Evaluation datasets loaded:",
        len(evaluation_data)
    )

    validation_df = pd.read_csv(
        VALIDATION_DIR / "normal.csv"
    )

    all_results = []

    total_models = (
        len(DATASETS)
        * len(SEEDS)
    )

    model_number = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for dataset_id in DATASETS:

        print()
        print("=" * 70)
        print(
            f"Training composition: {dataset_id}"
        )
        print("=" * 70)

        train_df = pd.read_csv(
            TRAIN_DIR /
            f"{dataset_id}.csv"
        )

        for seed in SEEDS:

            model_number += 1

            print()
            print(
                f"Model {model_number}/{total_models}"
            )

            print(
                f"Dataset={dataset_id}, Seed={seed}"
            )

            # --------------------------------------------
            # TRAIN EXACTLY ONCE
            # --------------------------------------------

            model, scaler, validation_mse = (
                train_model(
                    train_df=train_df,
                    validation_df=validation_df,
                    seed=seed,
                )
            )

            # --------------------------------------------
            # SAVE MODEL
            # --------------------------------------------

            checkpoint_path = save_checkpoint(
                model=model,
                scaler=scaler,
                dataset_id=dataset_id,
                seed=seed,
                validation_mse=validation_mse,
            )

            # --------------------------------------------
            # EVALUATE SAME MODEL EVERYWHERE
            # --------------------------------------------

            model_records = (
                evaluate_all_environments(
                    model=model,
                    scaler=scaler,
                    dataset_id=dataset_id,
                    seed=seed,
                    validation_mse=validation_mse,
                    evaluation_data=evaluation_data,
                )
            )

            all_results.extend(
                model_records
            )

            print(
                "Validation MSE:",
                f"{validation_mse:.6f}"
            )

            print(
                "Evaluations:",
                len(model_records)
            )

            print(
                "Checkpoint:",
                checkpoint_path
            )

    # ========================================================
    # CREATE MASTER TABLE
    # ========================================================

    master_df = pd.DataFrame(
        all_results
    )

    validate_results(
        master_df
    )

    master_file = (
        RESULT_DIR /
        "master_results.csv"
    )

    master_df.to_csv(
        master_file,
        index=False,
    )

    save_separate_results(
        master_df
    )

    save_config()

    print()
    print("=" * 70)
    print("CONFIRMATION EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        "Models trained:",
        total_models
    )

    print(
        "Total evaluations:",
        len(master_df)
    )

    print(
        "Master results:",
        master_file
    )

    print(
        "Checkpoints:",
        CHECKPOINT_DIR
    )

    print("=" * 70)


if __name__ == "__main__":
    main()