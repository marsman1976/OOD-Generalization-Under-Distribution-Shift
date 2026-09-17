from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.augmentation import apply_augmentation
from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 4 CONFIRMATION
#
# Research question:
#
# Does augmentation effectiveness depend on model
# configuration, and does this dependence change with
# distribution-shift severity?
#
# Frozen design:
#
# D00
# × A0, A1, A2, A3, A4
# × M0, M1, M2
# × 10 seeds
#
# = 150 trained models
#
# 20 evaluation environments/model
# = 3000 evaluation records
#
# IMPORTANT:
#
# This is the CONFIRMATORY Phase 4 experiment.
#
# DO NOT:
# - change augmentation definitions
# - change model architectures
# - change training hyperparameters
# - change seeds
# - change evaluation environments
# - stop based on performance or p-values
#
# Primary interaction:
#
# A × M × V
#
# with D fixed to D00.
# ============================================================


# ============================================================
# Project paths
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
    / "confirmation"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


RESULTS_FILE = (
    OUTPUT
    / "master_results.csv"
)


CONFIG_FILE = (
    OUTPUT
    / "confirmation_config.json"
)


# ============================================================
# Frozen Phase 4 factors
# ============================================================

DATASETS = [
    "D00",
]


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
    44,
    55,
    66,
    77,
    88,
    99,
    111,
]


# ============================================================
# Frozen augmentation seed convention
#
# Same convention used in Phase 2 and Phase 4 pilot.
# ============================================================

AUGMENTATION_SEED_OFFSET = 100000


# ============================================================
# Frozen model parameter counts
# ============================================================

MODEL_PARAMETERS = {

    "M1": 289,

    "M0": 833,

    "M2": 3201,

}


# ============================================================
# Frozen training protocol
# ============================================================

EPOCHS = 150

BATCH_SIZE = 64

LEARNING_RATE = 0.001

AUGMENTATION_FRACTION = 0.50


# ============================================================
# Evaluation environments
#
# EXACT SAME 20-environment structure as previous phases.
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

        "experiment":
            "EXP1",

        "environment":
            env,

        "shift_type":
            env,

        "severity":
            None,

        "shift_value":
            None,

        "path":
            TEST_DIR
            / f"{env}.csv",

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

        "experiment":
            "EXP1B",

        "environment":
            env,

        "shift_type":
            env,

        "severity":
            None,

        "shift_value":
            None,

        "path":
            OOD_DIR
            / f"{env}.csv",

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

        "experiment":
            "EXP1C",

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

        "experiment":
            "EXP1C",

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


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

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


# ============================================================
# Validation dataset
#
# Same validation environment as previous phases.
# ============================================================

validation_path = (
    VALIDATION_DIR
    / "normal.csv"
)


if not validation_path.exists():

    raise FileNotFoundError(
        validation_path
    )


validation_df = pd.read_csv(
    validation_path
)


# ============================================================
# Frozen confirmation configuration
# ============================================================

CONFIG = {

    "phase":
        "PHASE4",

    "run_type":
        "CONFIRMATION",

    "datasets":
        DATASETS,

    "augmentation_ids":
        AUGMENTATION_IDS,

    "model_configs":
        MODEL_CONFIGS,

    "seeds":
        SEEDS,

    "augmentation_seed_rule":
        (
            "augmentation_seed = "
            "100000 + training_seed"
        ),

    "augmentation_fraction":
        AUGMENTATION_FRACTION,

    "model_parameters":
        MODEL_PARAMETERS,

    "training_protocol":
        {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "optimizer":
                "Adam",

            "loss":
                "MSE",

            "scaler":
                "StandardScaler train-only",

            "validation":
                "normal.csv",

        },

    "n_models":
        150,

    "n_environments":
        20,

    "expected_records":
        3000,

    "primary_interaction":
        "A x M x V",

    "purpose":
        (
            "Phase 4 confirmation for "
            "augmentation x model configuration "
            "x shift severity with D fixed to D00"
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
        .groupby(
            "run_id"
        )
        .size()

    )

    return set(

        counts[
            counts == 20
        ].index

    )


# ============================================================
# Main confirmation experiment
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "PHASE 4 CONFIRMATION"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # Expected design size
    # ========================================================

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


    assert (
        expected_models
        == 150
    )


    assert (
        expected_records
        == 3000
    )


    print(
        "Datasets:",
        DATASETS
    )

    print(
        "Augmentations:",
        AUGMENTATION_IDS
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
        "Expected models:",
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
    # Save frozen confirmation configuration
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
    # Resume support
    # ========================================================

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
    # TRAINING LOOP
    # ========================================================

    for dataset_id in DATASETS:


        # ----------------------------------------------------
        # Load original D00 training data
        # ----------------------------------------------------

        train_path = (

            TRAIN_DIR

            / f"{dataset_id}.csv"

        )


        if not train_path.exists():

            raise FileNotFoundError(
                train_path
            )


        original_train_df = pd.read_csv(
            train_path
        )


        # ----------------------------------------------------
        # Frozen dataset integrity
        # ----------------------------------------------------

        assert (
            len(original_train_df)
            == 1000
        )


        assert not (

            original_train_df
            .isna()
            .any()
            .any()

        )


        # ====================================================
        # TRAINING SEEDS
        # ====================================================

        for seed in SEEDS:


            # ------------------------------------------------
            # Frozen augmentation seed
            # ------------------------------------------------

            augmentation_seed = (

                AUGMENTATION_SEED_OFFSET

                + seed

            )


            # ================================================
            # AUGMENTATION CONDITIONS
            # ================================================

            for augmentation_id in AUGMENTATION_IDS:


                # --------------------------------------------
                # Apply augmentation ONCE for this:
                #
                # D × A × seed
                #
                # Then use exactly the same augmented
                # dataframe for M0, M1 and M2.
                #
                # This is essential for the paired A × M
                # experimental design.
                # --------------------------------------------

                (
                    augmented_train_df,
                    aug_metadata,

                ) = apply_augmentation(

                    df=original_train_df,

                    augmentation_id=
                        augmentation_id,

                    seed=
                        augmentation_seed,

                    fraction=
                        AUGMENTATION_FRACTION,

                )


                # ============================================
                # AUGMENTATION INTEGRITY
                # ============================================

                if (
                    len(augmented_train_df)
                    != 1000
                ):

                    raise RuntimeError(

                        "Augmentation changed "
                        "training-set size: "

                        f"{dataset_id}, "
                        f"{augmentation_id}, "
                        f"seed={seed}"

                    )


                if (

                    augmented_train_df
                    .isna()
                    .any()
                    .any()

                ):

                    raise RuntimeError(

                        "Augmentation produced "
                        "missing values: "

                        f"{dataset_id}, "
                        f"{augmentation_id}, "
                        f"seed={seed}"

                    )


                expected_n_augmented = (

                    0

                    if augmentation_id
                    == "A0"

                    else 500

                )


                actual_n_augmented = int(

                    aug_metadata[
                        "n_augmented"
                    ]

                )


                if (

                    actual_n_augmented
                    != expected_n_augmented

                ):

                    raise RuntimeError(

                        "Unexpected augmentation "
                        "count for "

                        f"{augmentation_id}: "

                        f"{actual_n_augmented} "

                        f"(expected "
                        f"{expected_n_augmented})"

                    )


                # ============================================
                # MODEL CONFIGURATIONS
                # ============================================

                for model_config in MODEL_CONFIGS:


                    run_id = (

                        f"{dataset_id}_"

                        f"{augmentation_id}_"

                        f"{model_config}_"

                        f"seed{seed}"

                    )


                    # ========================================
                    # Resume support
                    # ========================================

                    if run_id in completed:

                        print(

                            f"SKIP complete: "
                            f"{run_id}"

                        )

                        continue


                    print()

                    print(
                        "-" * 78
                    )


                    print(
                        "Training:",
                        run_id
                    )


                    print(

                        f"Dataset="
                        f"{dataset_id}, "

                        f"Augmentation="
                        f"{augmentation_id}, "

                        f"Model="
                        f"{model_config}, "

                        f"Seed="
                        f"{seed}, "

                        f"AugSeed="
                        f"{augmentation_seed}"

                    )


                    # ========================================
                    # TRAIN MODEL
                    #
                    # Uses frozen Phase 3 training pipeline.
                    # ========================================

                    (
                        model,
                        scaler,
                        val_loss,

                    ) = train_phase3_model(

                        train_df=
                            augmented_train_df,

                        validation_df=
                            validation_df,

                        seed=
                            seed,

                        model_id=
                            model_config,

                        epochs=
                            EPOCHS,

                        batch_size=
                            BATCH_SIZE,

                        learning_rate=
                            LEARNING_RATE,

                    )


                    # ========================================
                    # Validation-loss integrity
                    # ========================================

                    if not np.isfinite(
                        val_loss
                    ):

                        raise RuntimeError(

                            "Non-finite validation "
                            "loss: "

                            f"{run_id}"

                        )


                    # ========================================
                    # EVALUATE SAME MODEL ON ALL 20
                    # ENVIRONMENTS
                    # ========================================

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


                        # ------------------------------------
                        # Numerical safety
                        # ------------------------------------

                        if not np.isfinite(
                            metrics["mse"]
                        ):

                            raise RuntimeError(

                                "Non-finite MSE: "

                                f"{run_id} "

                                f"{info['environment']}"

                            )


                        if not np.isfinite(
                            metrics["mae"]
                        ):

                            raise RuntimeError(

                                "Non-finite MAE: "

                                f"{run_id} "

                                f"{info['environment']}"

                            )


                        # ------------------------------------
                        # Save evaluation record
                        # ------------------------------------

                        model_records.append({

                            "phase":
                                "PHASE4",

                            "run_type":
                                "CONFIRMATION",

                            "run_id":
                                run_id,

                            "dataset_id":
                                dataset_id,

                            "augmentation_id":
                                augmentation_id,

                            "augmentation_seed":
                                augmentation_seed,

                            "augmentation_fraction":
                                float(
                                    aug_metadata[
                                        "fraction"
                                    ]
                                ),

                            "n_augmented":
                                actual_n_augmented,

                            "model_config":
                                model_config,

                            "model_parameters":
                                MODEL_PARAMETERS[
                                    model_config
                                ],

                            "seed":
                                seed,

                            "n_train":
                                len(
                                    augmented_train_df
                                ),

                            "experiment":
                                info[
                                    "experiment"
                                ],

                            "environment":
                                info[
                                    "environment"
                                ],

                            "shift_type":
                                info[
                                    "shift_type"
                                ],

                            "severity":
                                info[
                                    "severity"
                                ],

                            "shift_value":
                                info[
                                    "shift_value"
                                ],

                            "best_validation_loss":
                                val_loss,

                            "mse":
                                metrics[
                                    "mse"
                                ],

                            "mae":
                                metrics[
                                    "mae"
                                ],

                        })


                    # ========================================
                    # Every trained model must have exactly
                    # 20 evaluation records.
                    # ========================================

                    assert (
                        len(model_records)
                        == 20
                    )


                    new_records.extend(
                        model_records
                    )


                    trained_now += 1


                    # ========================================
                    # SAVE AFTER EVERY TRAINED MODEL
                    #
                    # Important for long confirmation runs.
                    # ========================================

                    new_df = pd.DataFrame(
                        new_records
                    )


                    if existing.empty:

                        combined = (
                            new_df
                        )

                    else:

                        combined = pd.concat(

                            [
                                existing,
                                new_df,
                            ],

                            ignore_index=True,

                        )


                    # ----------------------------------------
                    # Prevent duplicate evaluations
                    # ----------------------------------------

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
    # FINAL CONFIRMATION INTEGRITY CHECK
    # ========================================================

    final = pd.read_csv(
        RESULTS_FILE
    )


    print()

    print(
        "=" * 78
    )

    print(
        "PHASE 4 CONFIRMATION "
        "FINAL INTEGRITY CHECK"
    )

    print(
        "=" * 78
    )


    # ========================================================
    # Basic counts
    # ========================================================

    print(
        "Records:",
        len(final)
    )


    print(

        "Unique models:",

        final[
            "run_id"
        ].nunique()

    )


    # ========================================================
    # Duplicate check
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
    # 20 environments/model check
    # ========================================================

    counts = (

        final

        .groupby(
            "run_id"
        )

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
    # Experiment counts
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


    # --------------------------------------------------------
    # Expected confirmation counts
    #
    # 150 models
    #
    # EXP1:
    # 150 × 4 = 600
    #
    # EXP1B:
    # 150 × 3 = 450
    #
    # EXP1C:
    # 150 × 13 = 1950
    #
    # Total:
    # 3000
    # --------------------------------------------------------

    assert (
        len(final)
        == 3000
    )


    assert (

        final[
            "run_id"
        ].nunique()

        == 150

    )


    assert (
        duplicates
        == 0
    )


    assert (
        len(incomplete)
        == 0
    )


    assert (

        experiment_counts[
            "EXP1"
        ]

        == 600

    )


    assert (

        experiment_counts[
            "EXP1B"
        ]

        == 450

    )


    assert (

        experiment_counts[
            "EXP1C"
        ]

        == 1950

    )


    # ========================================================
    # Factor integrity
    # ========================================================

    assert set(

        final[
            "dataset_id"
        ].unique()

    ) == {
        "D00"
    }


    assert set(

        final[
            "augmentation_id"
        ].unique()

    ) == set(
        AUGMENTATION_IDS
    )


    assert set(

        final[
            "model_config"
        ].unique()

    ) == set(
        MODEL_CONFIGS
    )


    assert set(

        final[
            "seed"
        ].unique()

    ) == set(
        SEEDS
    )


    # ========================================================
    # Every A × M cell:
    #
    # 10 seeds × 20 environments = 200 records
    # ========================================================

    am_counts = (

        final

        .groupby([

            "augmentation_id",

            "model_config",

        ])

        .size()

    )


    assert (

        am_counts
        == 200

    ).all()


    # ========================================================
    # Every A × M cell must contain all 10 seeds
    # ========================================================

    am_seed_counts = (

        final

        .groupby([

            "augmentation_id",

            "model_config",

        ])[

            "seed"

        ]

        .nunique()

    )


    assert (

        am_seed_counts
        == 10

    ).all()


    # ========================================================
    # Parameter-count integrity
    # ========================================================

    for (
        model_config,
        expected_params,
    ) in MODEL_PARAMETERS.items():


        observed = set(

            final.loc[

                final[
                    "model_config"
                ]
                == model_config,

                "model_parameters",

            ]

            .astype(int)

        )


        assert observed == {
            expected_params
        }


    # ========================================================
    # Augmentation integrity
    # ========================================================

    a0_n_augmented = set(

        final.loc[

            final[
                "augmentation_id"
            ]
            == "A0",

            "n_augmented",

        ]

        .astype(int)

    )


    assert (
        a0_n_augmented
        == {0}
    )


    non_a0_n_augmented = set(

        final.loc[

            final[
                "augmentation_id"
            ]
            != "A0",

            "n_augmented",

        ]

        .astype(int)

    )


    assert (
        non_a0_n_augmented
        == {500}
    )


    # ========================================================
    # Augmentation-seed integrity
    # ========================================================

    for seed in SEEDS:


        observed_aug_seeds = set(

            final.loc[

                final[
                    "seed"
                ]
                == seed,

                "augmentation_seed",

            ]

            .astype(int)

        )


        expected_aug_seed = (

            AUGMENTATION_SEED_OFFSET

            + seed

        )


        assert (
            observed_aug_seeds
            == {
                expected_aug_seed
            }
        )


    # ========================================================
    # Training-set size
    # ========================================================

    assert set(

        final[
            "n_train"
        ]

        .astype(int)

        .unique()

    ) == {
        1000
    }


    # ========================================================
    # Numerical integrity
    # ========================================================

    for column in [

        "best_validation_loss",

        "mse",

        "mae",

    ]:


        assert (

            final[
                column
            ]

            .notna()

            .all()

        )


        assert np.isfinite(

            final[
                column
            ]

        ).all()


    # ========================================================
    # Environment integrity
    #
    # Every model must contain the same exact 20 environments.
    # ========================================================

    expected_environment_names = {

        info[
            "environment"
        ]

        for info in ENVIRONMENTS

    }


    for (
        run_id,
        group,
    ) in final.groupby(
        "run_id"
    ):


        observed_environments = set(

            group[
                "environment"
            ]

        )


        assert (

            observed_environments
            == expected_environment_names

        ), (

            f"Environment mismatch "
            f"for {run_id}"

        )


    # ========================================================
    # Severity structure integrity
    # ========================================================

    temp = final[

        (
            final[
                "experiment"
            ]
            == "EXP1C"
        )

        &

        (
            final[
                "shift_type"
            ]
            == "temperature"
        )

    ]


    humidity = final[

        (
            final[
                "experiment"
            ]
            == "EXP1C"
        )

        &

        (
            final[
                "shift_type"
            ]
            == "humidity"
        )

    ]


    assert set(

        temp[
            "severity"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == {
        0,
        1,
        2,
        3,
        4,
        5,
    }


    assert set(

        temp[
            "shift_value"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == set(
        TEMPERATURE_LEVELS
    )


    assert set(

        humidity[
            "severity"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    }


    assert set(

        humidity[
            "shift_value"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == set(
        HUMIDITY_LEVELS
    )


    # ========================================================
    # Exact EXP1C counts
    #
    # Temperature:
    # 150 × 6 = 900
    #
    # Humidity:
    # 150 × 7 = 1050
    # ========================================================

    assert (
        len(temp)
        == 900
    )


    assert (
        len(humidity)
        == 1050
    )


    # ========================================================
    # Runtime
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

    print(
        "=" * 78
    )

    print(
        "PHASE 4 CONFIRMATION: PASS"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()