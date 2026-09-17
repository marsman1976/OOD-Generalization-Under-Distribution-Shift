from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.augmentation import apply_augmentation
from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 4 ENGINEERING PILOT
#
# D00
# × A0, A1, A2, A3, A4
# × M0, M1, M2
# × seeds 11, 22, 33
#
# = 45 trained models
#
# 20 evaluations/model
# = 900 evaluation records
#
# IMPORTANT:
# This is an ENGINEERING pilot.
# Do not use pilot OOD performance to redesign/select A or M.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
TRAIN_DIR = DATA / "training"
VALIDATION_DIR = DATA / "validation"
TEST_DIR = DATA / "test"
OOD_DIR = DATA / "ood_test"
SEVERITY_DIR = DATA / "severity_test"

OUTPUT = (
    ROOT
    / "results"
    / "experiment_4"
    / "pilot"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUTPUT / "master_results.csv"
CONFIG_FILE = OUTPUT / "pilot_config.json"


# ============================================================
# Frozen Phase 4 pilot factors
# ============================================================

DATASETS = ["D00"]

AUGMENTATION_IDS = [
    "A0",
    "A1",
    "A2",
    "A3",
    "A4",
]

MODEL_CONFIGS = [
    "M0",
    "M1",
    "M2",
]

SEEDS = [
    11,
    22,
    33,
]

# Frozen Phase 2 augmentation-seed convention.
AUGMENTATION_SEED_OFFSET = 100000

MODEL_PARAMETERS = {
    "M1": 289,
    "M0": 833,
    "M2": 3201,
}


# ============================================================
# Evaluation environments
# EXACT SAME 20-environment structure as Phases 2/3.
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

# EXP1C — temperature severity
TEMPERATURE_LEVELS = [
    23,
    25,
    27,
    29,
    31,
    33,
]

for severity, value in enumerate(TEMPERATURE_LEVELS):
    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": f"temp_s{severity}",
        "shift_type": "temperature",
        "severity": severity,
        "shift_value": value,
        "path": SEVERITY_DIR / f"temp_s{severity}.csv",
    })

# EXP1C — humidity severity
HUMIDITY_LEVELS = [
    65,
    70,
    75,
    80,
    85,
    90,
    92,
]

for severity, value in enumerate(HUMIDITY_LEVELS):
    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": f"humidity_s{severity}",
        "shift_type": "humidity",
        "severity": severity,
        "shift_value": value,
        "path": SEVERITY_DIR / f"humidity_s{severity}.csv",
    })

assert len(ENVIRONMENTS) == 20


# ============================================================
# Preload evaluation + validation data
# ============================================================

evaluation_data = {}

for info in ENVIRONMENTS:
    if not info["path"].exists():
        raise FileNotFoundError(info["path"])

    evaluation_data[info["environment"]] = pd.read_csv(
        info["path"]
    )

validation_path = VALIDATION_DIR / "normal.csv"

if not validation_path.exists():
    raise FileNotFoundError(validation_path)

validation_df = pd.read_csv(validation_path)


# ============================================================
# Frozen configuration record
# ============================================================

CONFIG = {
    "phase": "PHASE4",
    "run_type": "ENGINEERING_PILOT",
    "datasets": DATASETS,
    "augmentation_ids": AUGMENTATION_IDS,
    "model_configs": MODEL_CONFIGS,
    "seeds": SEEDS,
    "augmentation_seed_rule": (
        "augmentation_seed = 100000 + training_seed"
    ),
    "augmentation_fraction": 0.50,
    "model_parameters": MODEL_PARAMETERS,
    "training_protocol": {
        "epochs": 150,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "loss": "MSE",
        "scaler": "StandardScaler train-only",
        "validation": "normal.csv",
    },
    "n_models": 45,
    "n_environments": 20,
    "expected_records": 900,
    "purpose": (
        "Phase 4 engineering pilot for A x M x S x V "
        "with D fixed to D00"
    ),
}


# ============================================================
# Resume support
# ============================================================

def load_existing():
    if not RESULTS_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(RESULTS_FILE)

    print(
        f"Existing records found: {len(df)}"
    )

    return df


def completed_model_ids(df):
    if df.empty:
        return set()

    counts = (
        df
        .groupby("run_id")
        .size()
    )

    return set(
        counts[counts == 20].index
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 76)
    print("PHASE 4 ENGINEERING PILOT")
    print("=" * 76)

    expected_models = (
        len(DATASETS)
        * len(AUGMENTATION_IDS)
        * len(MODEL_CONFIGS)
        * len(SEEDS)
    )

    expected_records = (
        expected_models
        * len(ENVIRONMENTS)
    )

    assert expected_models == 45
    assert expected_records == 900

    print("Datasets:", DATASETS)
    print("Augmentations:", AUGMENTATION_IDS)
    print("Model configurations:", MODEL_CONFIGS)
    print("Seeds:", SEEDS)
    print("Models:", expected_models)
    print("Environments/model:", len(ENVIRONMENTS))
    print("Expected records:", expected_records)

    # Save frozen pilot configuration.
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
    completed = completed_model_ids(existing)

    print(
        "Completed models already present:",
        len(completed)
    )

    new_records = []
    trained_now = 0
    start = time.time()

    # ========================================================
    # TRAINING
    # ========================================================

    for dataset_id in DATASETS:

        train_path = (
            TRAIN_DIR
            / f"{dataset_id}.csv"
        )

        if not train_path.exists():
            raise FileNotFoundError(train_path)

        original_train_df = pd.read_csv(
            train_path
        )

        assert len(original_train_df) == 1000
        assert not original_train_df.isna().any().any()

        for seed in SEEDS:

            # IMPORTANT:
            # Same frozen augmentation seed is used for all
            # model configurations at this training seed.
            augmentation_seed = (
                AUGMENTATION_SEED_OFFSET
                + seed
            )

            for augmentation_id in AUGMENTATION_IDS:

                # --------------------------------------------
                # Apply the frozen Phase 2 augmentation ONCE
                # for this D × A × seed combination.
                #
                # The resulting training dataframe is then
                # shared by M0/M1/M2, giving a clean paired
                # A × M comparison.
                # --------------------------------------------

                augmented_train_df, aug_metadata = (
                    apply_augmentation(
                        df=original_train_df,
                        augmentation_id=augmentation_id,
                        seed=augmentation_seed,
                        fraction=0.50,
                    )
                )

                if len(augmented_train_df) != 1000:
                    raise RuntimeError(
                        "Augmentation changed training-set size: "
                        f"{dataset_id}, {augmentation_id}, "
                        f"seed={seed}"
                    )

                if augmented_train_df.isna().any().any():
                    raise RuntimeError(
                        "Augmentation produced missing values: "
                        f"{dataset_id}, {augmentation_id}, "
                        f"seed={seed}"
                    )

                expected_n_augmented = (
                    0
                    if augmentation_id == "A0"
                    else 500
                )

                if (
                    int(aug_metadata["n_augmented"])
                    != expected_n_augmented
                ):
                    raise RuntimeError(
                        "Unexpected augmentation count for "
                        f"{augmentation_id}: "
                        f"{aug_metadata['n_augmented']} "
                        f"(expected {expected_n_augmented})"
                    )

                for model_config in MODEL_CONFIGS:

                    run_id = (
                        f"{dataset_id}_"
                        f"{augmentation_id}_"
                        f"{model_config}_"
                        f"seed{seed}"
                    )

                    # Resume support.
                    if run_id in completed:
                        print(
                            f"SKIP complete: {run_id}"
                        )
                        continue

                    print()
                    print("-" * 76)
                    print("Training:", run_id)
                    print(
                        f"Dataset={dataset_id}, "
                        f"Augmentation={augmentation_id}, "
                        f"Model={model_config}, "
                        f"Seed={seed}, "
                        f"AugSeed={augmentation_seed}"
                    )

                    # ========================================
                    # TRAINING
                    # ========================================

                    model, scaler, val_loss = (
                        train_phase3_model(
                            train_df=augmented_train_df,
                            validation_df=validation_df,
                            seed=seed,
                            model_id=model_config,
                        )
                    )

                    if not np.isfinite(val_loss):
                        raise RuntimeError(
                            f"Non-finite validation loss: {run_id}"
                        )

                    # ========================================
                    # EVALUATION
                    # Same fitted model on all 20 environments.
                    # ========================================

                    model_records = []

                    for info in ENVIRONMENTS:

                        test_df = evaluation_data[
                            info["environment"]
                        ]

                        metrics = evaluate_model(
                            model,
                            scaler,
                            test_df,
                        )

                        if not np.isfinite(metrics["mse"]):
                            raise RuntimeError(
                                f"Non-finite MSE: {run_id} "
                                f"{info['environment']}"
                            )

                        if not np.isfinite(metrics["mae"]):
                            raise RuntimeError(
                                f"Non-finite MAE: {run_id} "
                                f"{info['environment']}"
                            )

                        model_records.append({
                            "phase": "PHASE4",
                            "run_type": "ENGINEERING_PILOT",
                            "run_id": run_id,
                            "dataset_id": dataset_id,
                            "augmentation_id": augmentation_id,
                            "augmentation_seed": augmentation_seed,
                            "augmentation_fraction": (
                                float(aug_metadata["fraction"])
                            ),
                            "n_augmented": (
                                int(aug_metadata["n_augmented"])
                            ),
                            "model_config": model_config,
                            "model_parameters": (
                                MODEL_PARAMETERS[model_config]
                            ),
                            "seed": seed,
                            "n_train": len(
                                augmented_train_df
                            ),
                            "experiment": info["experiment"],
                            "environment": info["environment"],
                            "shift_type": info["shift_type"],
                            "severity": info["severity"],
                            "shift_value": info["shift_value"],
                            "best_validation_loss": val_loss,
                            "mse": metrics["mse"],
                            "mae": metrics["mae"],
                        })

                    assert len(model_records) == 20

                    new_records.extend(
                        model_records
                    )

                    trained_now += 1

                    # ========================================
                    # SAVE AFTER EVERY MODEL
                    # ========================================

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

                    combined = (
                        combined
                        .drop_duplicates(
                            subset=[
                                "run_id",
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
    # FINAL PILOT INTEGRITY
    # ========================================================

    final = pd.read_csv(
        RESULTS_FILE
    )

    print()
    print("=" * 76)
    print(
        "PHASE 4 PILOT FINAL INTEGRITY CHECK"
    )
    print("=" * 76)

    print("Records:", len(final))
    print(
        "Unique models:",
        final["run_id"].nunique()
    )

    duplicates = (
        final
        .duplicated(
            subset=[
                "run_id",
                "experiment",
                "environment",
            ]
        )
        .sum()
    )

    print(
        "Duplicate records:",
        duplicates
    )

    counts = (
        final
        .groupby("run_id")
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
    # EXP1  = 45 × 4  = 180
    # EXP1B = 45 × 3  = 135
    # EXP1C = 45 × 13 = 585
    # Total = 900

    assert len(final) == 900

    assert (
        final["run_id"].nunique()
        == 45
    )

    assert duplicates == 0
    assert len(incomplete) == 0

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

    # Factor integrity.
    assert set(
        final["dataset_id"].unique()
    ) == {"D00"}

    assert set(
        final["augmentation_id"].unique()
    ) == set(AUGMENTATION_IDS)

    assert set(
        final["model_config"].unique()
    ) == set(MODEL_CONFIGS)

    assert set(
        final["seed"].unique()
    ) == set(SEEDS)

    # Each A × M combination must have 3 seeds × 20 env = 60 rows.
    am_counts = (
        final
        .groupby([
            "augmentation_id",
            "model_config",
        ])
        .size()
    )

    assert (
        am_counts == 60
    ).all()

    # Parameter-count integrity.
    for model_config, expected_params in MODEL_PARAMETERS.items():

        observed = set(
            final.loc[
                final["model_config"]
                == model_config,
                "model_parameters",
            ].astype(int)
        )

        assert observed == {
            expected_params
        }

    # Augmentation integrity.
    a0_n_aug = set(
        final.loc[
            final["augmentation_id"] == "A0",
            "n_augmented",
        ].astype(int)
    )

    assert a0_n_aug == {0}

    non_a0_n_aug = set(
        final.loc[
            final["augmentation_id"] != "A0",
            "n_augmented",
        ].astype(int)
    )

    assert non_a0_n_aug == {500}

    # Same augmentation seed for every A/M at a training seed.
    for seed in SEEDS:
        observed_aug_seeds = set(
            final.loc[
                final["seed"] == seed,
                "augmentation_seed",
            ].astype(int)
        )

        assert observed_aug_seeds == {
            AUGMENTATION_SEED_OFFSET + seed
        }

    # Numerical integrity.
    for column in [
        "best_validation_loss",
        "mse",
        "mae",
    ]:
        assert final[column].notna().all()
        assert np.isfinite(
            final[column]
        ).all()

    elapsed = time.time() - start

    print()
    print(
        f"Runtime this session: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print("=" * 76)
    print(
        "PHASE 4 ENGINEERING PILOT: PASS"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()