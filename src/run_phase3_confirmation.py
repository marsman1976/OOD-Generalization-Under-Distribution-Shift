from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 3 CONFIRMATION
#
# D00-D09 × M0-M2 × 10 seeds
#
# 10 × 3 × 10 = 300 trained models
#
# 20 evaluations/model
#
# 300 × 20 = 6,000 evaluation records
#
# A = A0 throughout Phase 3.
#
# FROZEN CONFIRMATORY EXPERIMENT
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
    / "confirmation"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_FILE = (
    OUTPUT / "master_results.csv"
)

CONFIG_FILE = (
    OUTPUT / "confirmation_config.json"
)


# ============================================================
# FROZEN EXPERIMENTAL FACTORS
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


MODEL_CONFIGS = [
    "M0",
    "M1",
    "M2",
]


SEEDS = [
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    111,
]


# ============================================================
# MODEL PARAMETER COUNTS
# ============================================================


MODEL_PARAMETERS = {
    "M1": 289,
    "M0": 833,
    "M2": 3201,
}


# ============================================================
# EVALUATION ENVIRONMENTS
#
# EXACT SAME 20 ENVIRONMENTS AS PHASES 1/2
# AND THE PHASE 3 ENGINEERING PILOT.
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
# EXP1C — TEMPERATURE SEVERITY
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
# EXP1C — HUMIDITY SEVERITY
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


# There must always be exactly 20 environments.

assert len(ENVIRONMENTS) == 20


# ============================================================
# PRELOAD EVALUATION DATA
# ============================================================


evaluation_data = {}


for info in ENVIRONMENTS:

    if not info["path"].exists():

        raise FileNotFoundError(
            info["path"]
        )

    df = pd.read_csv(
        info["path"]
    )

    if df.isna().any().any():

        raise RuntimeError(
            f"Missing values found in "
            f"{info['path']}"
        )

    evaluation_data[
        info["environment"]
    ] = df


# ============================================================
# VALIDATION DATA
#
# SAME NORMAL VALIDATION ENVIRONMENT AS BEFORE
# ============================================================


validation_path = (
    VALIDATION_DIR / "normal.csv"
)


if not validation_path.exists():

    raise FileNotFoundError(
        validation_path
    )


validation_df = pd.read_csv(
    validation_path
)


# ============================================================
# CONFIRMATION CONFIGURATION
# ============================================================


CONFIG = {

    "phase":
        "PHASE3",

    "run_type":
        "CONFIRMATION",

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
        300,

    "n_environments":
        20,

    "expected_records":
        6000,

    "purpose":
        (
            "Confirmatory Phase 3 "
            "D x M x S x V experiment"
        ),
}


# ============================================================
# RESUME SUPPORT
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


def completed_run_ids(df):

    if df.empty:

        return set()

    counts = (
        df
        .groupby("run_id")
        .size()
    )

    # A model/run is complete only when
    # all 20 environments are present.

    return set(
        counts[
            counts == 20
        ].index
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 76)

    print(
        "PHASE 3 CONFIRMATION"
    )

    print("=" * 76)


    # ========================================================
    # EXPECTED EXPERIMENT SIZE
    # ========================================================


    expected_models = (
        len(DATASETS)
        * len(MODEL_CONFIGS)
        * len(SEEDS)
    )


    expected_records = (
        expected_models
        * len(ENVIRONMENTS)
    )


    assert expected_models == 300

    assert expected_records == 6000


    print(
        "Datasets:",
        len(DATASETS)
    )

    print(
        "Model configurations:",
        len(MODEL_CONFIGS)
    )

    print(
        "Seeds:",
        len(SEEDS)
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


    # ========================================================
    # SAVE FROZEN CONFIGURATION
    # ========================================================


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


    # ========================================================
    # LOAD PREVIOUS PROGRESS
    # ========================================================


    existing = load_existing()

    completed = completed_run_ids(
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
    # TRAINING LOOP
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


        # Frozen training-set size

        assert len(train_df) == 1000


        if train_df.isna().any().any():

            raise RuntimeError(
                f"Missing values in "
                f"{train_path}"
            )


        # ====================================================
        # SEEDS
        # ====================================================


        for seed in SEEDS:


            # ================================================
            # MODEL CONFIGURATION
            # ================================================


            for model_config in MODEL_CONFIGS:


                run_id = (
                    f"{dataset_id}_"
                    f"{model_config}_"
                    f"seed{seed}"
                )


                # ============================================
                # RESUME SUPPORT
                # ============================================


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
                # TRAIN MODEL
                #
                # AUGMENTATION IS FIXED AT A0.
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


                if not np.isfinite(
                    val_loss
                ):

                    raise RuntimeError(
                        f"Non-finite validation "
                        f"loss for {run_id}"
                    )


                # ============================================
                # EVALUATE SAME FITTED MODEL
                # ON ALL 20 ENVIRONMENTS
                # ============================================


                model_records = []


                for info in ENVIRONMENTS:


                    test_df = (
                        evaluation_data[
                            info[
                                "environment"
                            ]
                        ]
                    )


                    metrics = evaluate_model(
                        model,
                        scaler,
                        test_df,
                    )


                    mse = metrics["mse"]

                    mae = metrics["mae"]


                    # ========================================
                    # NUMERICAL SAFETY
                    # ========================================


                    if not np.isfinite(mse):

                        raise RuntimeError(
                            f"Non-finite MSE: "
                            f"{run_id} "
                            f"{info['environment']}"
                        )


                    if not np.isfinite(mae):

                        raise RuntimeError(
                            f"Non-finite MAE: "
                            f"{run_id} "
                            f"{info['environment']}"
                        )


                    # ========================================
                    # RECORD RESULT
                    # ========================================


                    model_records.append({

                        "phase":
                            "PHASE3",

                        "run_type":
                            "CONFIRMATION",

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
                            mse,

                        "mae":
                            mae,
                    })


                # Every model must have
                # exactly 20 evaluations.

                assert (
                    len(model_records)
                    == 20
                )


                new_records.extend(
                    model_records
                )


                trained_now += 1


                # ============================================
                # SAVE AFTER EVERY TRAINED MODEL
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


                # Remove accidental duplicate
                # model/environment records.

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
    # FINAL INTEGRITY AUDIT
    # ========================================================


    final = pd.read_csv(
        RESULTS_FILE
    )


    print()

    print("=" * 76)

    print(
        "PHASE 3 CONFIRMATION "
        "FINAL INTEGRITY CHECK"
    )

    print("=" * 76)


    # ========================================================
    # BASIC COUNTS
    # ========================================================


    print(
        "Records:",
        len(final)
    )


    print(
        "Unique models:",
        final["run_id"].nunique()
    )


    # ========================================================
    # DUPLICATES
    # ========================================================


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


    # ========================================================
    # INCOMPLETE MODELS
    # ========================================================


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


    # ========================================================
    # EXPERIMENT COUNTS
    # ========================================================


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


    # ========================================================
    # EXPECTED:
    #
    # EXP1:
    # 300 × 4 = 1200
    #
    # EXP1B:
    # 300 × 3 = 900
    #
    # EXP1C:
    # 300 × 13 = 3900
    #
    # TOTAL:
    # 6000
    # ========================================================


    assert len(final) == 6000


    assert (
        final["run_id"].nunique()
        == 300
    )


    assert duplicates == 0


    assert len(incomplete) == 0


    assert (
        experiment_counts["EXP1"]
        == 1200
    )


    assert (
        experiment_counts["EXP1B"]
        == 900
    )


    assert (
        experiment_counts["EXP1C"]
        == 3900
    )


    # ========================================================
    # FACTOR INTEGRITY
    # ========================================================


    assert set(
        final[
            "dataset_id"
        ].unique()
    ) == set(DATASETS)


    assert set(
        final[
            "model_config"
        ].unique()
    ) == set(MODEL_CONFIGS)


    assert set(
        final[
            "seed"
        ].unique()
    ) == set(SEEDS)


    assert set(
        final[
            "augmentation_id"
        ].unique()
    ) == {"A0"}


    # ========================================================
    # COMPLETE FACTORIAL GRID
    #
    # Every D × M × seed combination
    # must occur exactly once as a trained model.
    # ========================================================


    model_grid = (
        final[
            [
                "dataset_id",
                "model_config",
                "seed",
                "run_id",
            ]
        ]
        .drop_duplicates()
    )


    assert len(model_grid) == 300


    grid_counts = (
        model_grid
        .groupby(
            [
                "dataset_id",
                "model_config",
                "seed",
            ]
        )
        .size()
    )


    assert (
        grid_counts == 1
    ).all()


    # ========================================================
    # NUMERICAL INTEGRITY
    # ========================================================


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


    assert (
        final[
            "best_validation_loss"
        ]
        .notna()
        .all()
    )


    assert np.isfinite(
        final["mse"]
    ).all()


    assert np.isfinite(
        final["mae"]
    ).all()


    assert np.isfinite(
        final[
            "best_validation_loss"
        ]
    ).all()


    # ========================================================
    # MODEL PARAMETER INTEGRITY
    # ========================================================


    for model_config, expected_params in (
        MODEL_PARAMETERS.items()
    ):

        observed = set(
            final.loc[
                final["model_config"]
                == model_config,
                "model_parameters"
            ].unique()
        )

        assert observed == {
            expected_params
        }


    # ========================================================
    # ENVIRONMENT INTEGRITY
    # ========================================================


    expected_environment_names = {
        info["environment"]
        for info in ENVIRONMENTS
    }


    observed_environment_names = set(
        final[
            "environment"
        ].unique()
    )


    assert (
        observed_environment_names
        ==
        expected_environment_names
    )


    # ========================================================
    # EACH MODEL MUST HAVE SAME 20 ENVIRONMENTS
    # ========================================================


    for run_id, group in (
        final.groupby("run_id")
    ):

        observed = set(
            group[
                "environment"
            ]
        )

        assert (
            observed
            ==
            expected_environment_names
        )


    # ========================================================
    # FINISHED
    # ========================================================


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
        "PHASE 3 CONFIRMATION: PASS"
    )

    print("=" * 76)


if __name__ == "__main__":
    main()