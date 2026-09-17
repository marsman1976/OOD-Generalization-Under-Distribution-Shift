from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 3 ENGINEERING PILOT
#
# D00, D05, D07
# × M0, M1, M2
# × seeds 11, 22, 33
#
# = 27 trained models
#
# 20 evaluations/model
# = 540 evaluation records
#
# IMPORTANT:
# A = A0 throughout Phase 3.
#
# This is an ENGINEERING pilot.
# Do not use pilot OOD results to redesign M1/M2.
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
    / "experiment_3"
    / "pilot"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FILE = (
    OUTPUT / "master_results.csv"
)

CONFIG_FILE = (
    OUTPUT / "pilot_config.json"
)


# ============================================================
# Frozen Phase 3 pilot factors
# ============================================================

DATASETS = [
    "D00",
    "D05",
    "D07",
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


# ============================================================
# Model parameter counts
# ============================================================

MODEL_PARAMETERS = {
    "M1": 289,
    "M0": 833,
    "M2": 3201,
}


# ============================================================
# Evaluation environments
#
# EXACT SAME 20-environment structure as Phase 2.
# ============================================================

ENVIRONMENTS = []


# ------------------------------------------------------------
# EXP1
# ------------------------------------------------------------

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

        "path":
            TEST_DIR / f"{env}.csv",
    })


# ------------------------------------------------------------
# EXP1B
# ------------------------------------------------------------

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

        "path":
            OOD_DIR / f"{env}.csv",
    })


# ------------------------------------------------------------
# EXP1C — temperature severity
# ------------------------------------------------------------

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

    ENVIRONMENTS.append({

        "experiment": "EXP1C",

        "environment":
            f"temp_s{severity}",

        "shift_type":
            "temperature",

        "severity":
            severity,

        "shift_value":
            value,

        "path":
            SEVERITY_DIR
            / f"temp_s{severity}.csv",
    })


# ------------------------------------------------------------
# EXP1C — humidity severity
# ------------------------------------------------------------

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

    ENVIRONMENTS.append({

        "experiment": "EXP1C",

        "environment":
            f"humidity_s{severity}",

        "shift_type":
            "humidity",

        "severity":
            severity,

        "shift_value":
            value,

        "path":
            SEVERITY_DIR
            / f"humidity_s{severity}.csv",
    })


assert len(ENVIRONMENTS) == 20


# ============================================================
# Preload evaluation data
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


# Same validation environment as Phases 1/2

validation_df = pd.read_csv(
    VALIDATION_DIR / "normal.csv"
)


# ============================================================
# Frozen configuration record
# ============================================================

CONFIG = {

    "phase":
        "PHASE3",

    "run_type":
        "ENGINEERING_PILOT",

    "datasets":
        DATASETS,

    "model_configs":
        MODEL_CONFIGS,

    "seeds":
        SEEDS,

    "augmentation_id":
        "A0",

    "model_parameters":
        MODEL_PARAMETERS,

    "training_protocol":
        {
            "epochs": 150,
            "batch_size": 64,
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "loss": "MSE",
            "scaler":
                "StandardScaler train-only",
            "validation":
                "normal.csv",
        },

    "n_models":
        27,

    "n_environments":
        20,

    "expected_records":
        540,

    "purpose":
        (
            "Phase 3 engineering pilot "
            "for D x M x S x V"
        ),
}


# ============================================================
# Resume support
# ============================================================

def load_existing():

    if not RESULTS_FILE.exists():

        return pd.DataFrame()

    df = pd.read_csv(
        RESULTS_FILE
    )

    print(
        f"Existing records found: "
        f"{len(df)}"
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
        counts[
            counts == 20
        ].index
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 76)

    print(
        "PHASE 3 ENGINEERING PILOT"
    )

    print("=" * 76)

    expected_models = (
        len(DATASETS)
        * len(MODEL_CONFIGS)
        * len(SEEDS)
    )

    expected_records = (
        expected_models
        * len(ENVIRONMENTS)
    )

    assert expected_models == 27
    assert expected_records == 540

    print(
        "Datasets:",
        DATASETS
    )

    print(
        "Model configurations:",
        MODEL_CONFIGS
    )

    print(
        "Seeds:",
        SEEDS
    )

    print(
        "Models:",
        expected_models
    )

    print(
        "Environments/model:",
        len(ENVIRONMENTS)
    )

    print(
        "Expected records:",
        expected_records
    )


    # --------------------------------------------------------
    # Save frozen pilot configuration
    # --------------------------------------------------------

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
    # TRAINING
    # ========================================================

    for dataset_id in DATASETS:

        train_path = (
            TRAIN_DIR
            / f"{dataset_id}.csv"
        )

        if not train_path.exists():

            raise FileNotFoundError(
                train_path
            )

        train_df = pd.read_csv(
            train_path
        )

        assert len(train_df) == 1000

        assert not (
            train_df
            .isna()
            .any()
            .any()
        )


        for seed in SEEDS:


            for model_config in MODEL_CONFIGS:

                run_id = (
                    f"{dataset_id}_"
                    f"{model_config}_"
                    f"seed{seed}"
                )


                # --------------------------------------------
                # Resume support
                # --------------------------------------------

                if run_id in completed:

                    print(
                        f"SKIP complete: "
                        f"{run_id}"
                    )

                    continue


                print()

                print("-" * 76)

                print(
                    "Training:",
                    run_id
                )

                print(
                    f"Dataset={dataset_id}, "
                    f"Model={model_config}, "
                    f"Seed={seed}"
                )


                # ============================================
                # TRAINING
                #
                # A0 ONLY — no augmentation.
                # ============================================

                model, scaler, val_loss = (
                    train_phase3_model(
                        train_df=train_df,
                        validation_df=
                            validation_df,
                        seed=seed,
                        model_id=
                            model_config,
                    )
                )


                # ============================================
                # EVALUATION
                #
                # Evaluate SAME fitted model
                # on all 20 environments.
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


                    # Numerical safety checks

                    if not np.isfinite(
                        metrics["mse"]
                    ):

                        raise RuntimeError(
                            f"Non-finite MSE: "
                            f"{run_id} "
                            f"{info['environment']}"
                        )


                    if not np.isfinite(
                        metrics["mae"]
                    ):

                        raise RuntimeError(
                            f"Non-finite MAE: "
                            f"{run_id} "
                            f"{info['environment']}"
                        )


                    model_records.append({

                        "phase":
                            "PHASE3",

                        "run_type":
                            "ENGINEERING_PILOT",

                        "run_id":
                            run_id,

                        "dataset_id":
                            dataset_id,

                        "augmentation_id":
                            "A0",

                        "model_config":
                            model_config,

                        "model_parameters":
                            MODEL_PARAMETERS[
                                model_config
                            ],

                        "seed":
                            seed,

                        "n_train":
                            len(train_df),

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


                assert (
                    len(model_records)
                    == 20
                )


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
        "PHASE 3 PILOT "
        "FINAL INTEGRITY CHECK"
    )

    print("=" * 76)


    print(
        "Records:",
        len(final)
    )

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


    incomplete = (
        counts[
            counts != 20
        ]
    )


    print(
        "Incomplete models:",
        len(incomplete)
    )


    experiment_counts = (
        final[
            "experiment"
        ]
        .value_counts()
        .sort_index()
    )


    print()

    print(
        "Experiment counts:"
    )

    print(
        experiment_counts
    )


    # --------------------------------------------------------
    # Expected counts
    #
    # EXP1:
    # 27 × 4 = 108
    #
    # EXP1B:
    # 27 × 3 = 81
    #
    # EXP1C:
    # 27 × 13 = 351
    #
    # Total = 540
    # --------------------------------------------------------

    assert len(final) == 540

    assert (
        final["run_id"].nunique()
        == 27
    )

    assert duplicates == 0

    assert len(incomplete) == 0


    assert (
        experiment_counts["EXP1"]
        == 108
    )

    assert (
        experiment_counts["EXP1B"]
        == 81
    )

    assert (
        experiment_counts["EXP1C"]
        == 351
    )


    # --------------------------------------------------------
    # Factor integrity
    # --------------------------------------------------------

    assert set(
        final["dataset_id"].unique()
    ) == set(DATASETS)


    assert set(
        final["model_config"].unique()
    ) == set(MODEL_CONFIGS)


    assert set(
        final["seed"].unique()
    ) == set(SEEDS)


    assert set(
        final["augmentation_id"].unique()
    ) == {"A0"}


    assert (
        final["mse"]
        .notna()
        .all()
    )

    assert (
        final["mae"]
        .notna()
        .all()
    )

    assert np.isfinite(
        final["mse"]
    ).all()

    assert np.isfinite(
        final["mae"]
    ).all()


    elapsed = (
        time.time()
        - start
    )


    print()

    print(
        f"Runtime this session: "
        f"{elapsed / 60:.2f} minutes"
    )


    print()

    print("=" * 76)

    print(
        "PHASE 3 ENGINEERING "
        "PILOT: PASS"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()