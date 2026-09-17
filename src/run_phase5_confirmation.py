from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.augmentation import apply_augmentation
from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


# ============================================================
# PHASE 5 CONFIRMATION
#
# D00-D09
# × A0-A4
# × M0-M2
# × 10 independent confirmation seeds
#
# = 1500 trained models
#
# 20 evaluations/model
# = 30000 evaluation records
#
# IMPORTANT:
# This is the frozen Phase 5 confirmation experiment.
#
# Do NOT change:
# - training-data compositions
# - augmentation strategies
# - model configurations
# - confirmation seeds
# - shift environments
# - augmentation rules
# - training protocol
#
# based on observed confirmation performance.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

TRAIN_DIR = DATA / "training"
VALIDATION_DIR = DATA / "validation"
TEST_DIR = DATA / "test"
OOD_DIR = DATA / "ood_test"
SEVERITY_DIR = DATA / "severity_test"


# ============================================================
# Output
# ============================================================

OUTPUT = (
    ROOT
    / "results"
    / "experiment_5"
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
# Frozen Phase 5 confirmation factors
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


# ============================================================
# Independent Phase 5 confirmation seeds
#
# IMPORTANT:
# These do NOT overlap with the Phase 5 engineering-pilot
# seeds 11, 22, 33.
# ============================================================

SEEDS = [
    121,
    132,
    143,
    154,
    165,
    176,
    187,
    198,
    209,
    220,
]


# ============================================================
# Frozen Phase 2 augmentation-seed convention
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
# Evaluation environments
#
# EXACT SAME 20-environment structure used by the previous
# verified experiments.
# ============================================================

ENVIRONMENTS = []


# ============================================================
# EXP1
#
# 4 standard-domain evaluation environments
# ============================================================

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
        "path": (
            TEST_DIR
            / f"{env}.csv"
        ),
    })


# ============================================================
# EXP1B
#
# 3 held-out OOD environments
# ============================================================

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
        "path": (
            OOD_DIR
            / f"{env}.csv"
        ),
    })


# ============================================================
# EXP1C — temperature severity
#
# Files:
#
# temp_s0.csv -> 23
# temp_s1.csv -> 25
# temp_s2.csv -> 27
# temp_s3.csv -> 29
# temp_s4.csv -> 31
# temp_s5.csv -> 33
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

    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": (
            f"temp_s{severity}"
        ),
        "shift_type": "temperature",
        "severity": severity,
        "shift_value": value,
        "path": (
            SEVERITY_DIR
            / f"temp_s{severity}.csv"
        ),
    })


# ============================================================
# EXP1C — humidity severity
#
# Files:
#
# humidity_s0.csv -> 65
# humidity_s1.csv -> 70
# humidity_s2.csv -> 75
# humidity_s3.csv -> 80
# humidity_s4.csv -> 85
# humidity_s5.csv -> 90
# humidity_s6.csv -> 92
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

    ENVIRONMENTS.append({
        "experiment": "EXP1C",
        "environment": (
            f"humidity_s{severity}"
        ),
        "shift_type": "humidity",
        "severity": severity,
        "shift_value": value,
        "path": (
            SEVERITY_DIR
            / f"humidity_s{severity}.csv"
        ),
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


# ============================================================
# Validation data
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
# Frozen configuration record
# ============================================================

CONFIG = {

    "phase":
        "PHASE5",

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

    "pilot_seeds": [
        11,
        22,
        33,
    ],

    "pilot_seed_overlap":
        False,

    "augmentation_seed_rule":
        (
            "augmentation_seed = "
            "100000 + training_seed"
        ),

    "augmentation_fraction":
        0.50,

    "model_parameters":
        MODEL_PARAMETERS,

    "training_protocol": {

        "epochs":
            150,

        "batch_size":
            64,

        "learning_rate":
            0.001,

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
        1500,

    "n_environments":
        20,

    "expected_records":
        30000,

    "expected_exp1_records":
        6000,

    "expected_exp1b_records":
        4500,

    "expected_exp1c_records":
        19500,

    "expected_temperature_records":
        9000,

    "expected_humidity_records":
        10500,

    "purpose":
        (
            "Phase 5 frozen confirmation "
            "experiment for "
            "D x A x M x S x V"
        ),

    "primary_hypothesis":
        (
            "H5.5: trajectory-level "
            "D x A x M interaction"
        ),

    "key_secondary_hypothesis":
        (
            "H5.6: D x A x M interaction "
            "changes with shift severity V"
        ),

    "augmentation_sharing_rule":
        (
            "For each D x A x seed, "
            "augmentation is generated once "
            "and the exact augmented dataframe "
            "is shared across M0, M1, and M2."
        ),

    "stopping_rule":
        (
            "Run until all 1500 models have "
            "20 evaluations each, producing "
            "30000 valid evaluation records. "
            "No performance-based or "
            "p-value-based early stopping."
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

    print(
        "=" * 76
    )

    print(
        "PHASE 5 CONFIRMATION"
    )

    print(
        "=" * 76
    )


    # ========================================================
    # Expected experiment size
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


    assert expected_models == 1500
    assert expected_records == 30000


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
        "Confirmation seeds:",
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
    # Resume information
    # ========================================================

    existing = load_existing()

    completed = (
        completed_model_ids(
            existing
        )
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


        # ----------------------------------------------------
        # Load frozen training composition
        # ----------------------------------------------------

        train_path = (
            TRAIN_DIR
            / f"{dataset_id}.csv"
        )


        if not train_path.exists():

            raise FileNotFoundError(
                train_path
            )


        original_train_df = (
            pd.read_csv(
                train_path
            )
        )


        if len(
            original_train_df
        ) != 1000:

            raise RuntimeError(
                f"{dataset_id}: "
                f"expected 1000 training rows, "
                f"found "
                f"{len(original_train_df)}"
            )


        if (
            original_train_df
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                f"{dataset_id}: "
                "training data contains "
                "missing values."
            )


        # ====================================================
        # Confirmation seeds
        # ====================================================

        for seed in SEEDS:


            # ------------------------------------------------
            # Frozen augmentation seed
            # ------------------------------------------------

            augmentation_seed = (
                AUGMENTATION_SEED_OFFSET
                + seed
            )


            # =================================================
            # Augmentation
            # =================================================

            for augmentation_id in (
                AUGMENTATION_IDS
            ):


                # =============================================
                # CRITICAL CONTROL
                #
                # Augmentation is generated ONCE for:
                #
                #     D × A × seed
                #
                # The resulting dataframe is then shared
                # unchanged across:
                #
                #     M0
                #     M1
                #     M2
                #
                # =============================================

                (
                    augmented_train_df,
                    aug_metadata,
                ) = apply_augmentation(

                    df=original_train_df,

                    augmentation_id=(
                        augmentation_id
                    ),

                    seed=augmentation_seed,

                    fraction=0.50,
                )


                # =============================================
                # Augmentation integrity
                # =============================================

                if (
                    len(
                        augmented_train_df
                    )
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


                if (
                    int(
                        aug_metadata[
                            "n_augmented"
                        ]
                    )
                    != expected_n_augmented
                ):

                    raise RuntimeError(

                        "Unexpected augmentation "
                        "count for "

                        f"{augmentation_id}: "

                        f"{aug_metadata['n_augmented']} "

                        f"(expected "
                        f"{expected_n_augmented})"
                    )


                # =============================================
                # Model configurations
                # =============================================

                for model_config in (
                    MODEL_CONFIGS
                ):


                    run_id = (

                        f"{dataset_id}_"

                        f"{augmentation_id}_"

                        f"{model_config}_"

                        f"seed{seed}"
                    )


                    # =========================================
                    # Resume support
                    # =========================================

                    if run_id in completed:

                        print(
                            f"SKIP complete: "
                            f"{run_id}"
                        )

                        continue


                    print()

                    print(
                        "-" * 76
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


                    # =========================================
                    # TRAINING
                    # =========================================

                    (
                        model,
                        scaler,
                        val_loss,
                    ) = train_phase3_model(

                        train_df=(
                            augmented_train_df
                        ),

                        validation_df=(
                            validation_df
                        ),

                        seed=seed,

                        model_id=(
                            model_config
                        ),
                    )


                    if not np.isfinite(
                        val_loss
                    ):

                        raise RuntimeError(

                            "Non-finite validation "
                            "loss: "

                            f"{run_id}"
                        )


                    # =========================================
                    # EVALUATION
                    #
                    # Same fitted model evaluated on all
                    # 20 frozen environments.
                    # =========================================

                    model_records = []


                    for info in ENVIRONMENTS:


                        test_df = (

                            evaluation_data[
                                info[
                                    "environment"
                                ]
                            ]
                        )


                        metrics = (
                            evaluate_model(

                                model,

                                scaler,

                                test_df,
                            )
                        )


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


                        # =====================================
                        # Save evaluation record
                        # =====================================

                        model_records.append({

                            "phase":
                                "PHASE5",

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
                                int(
                                    aug_metadata[
                                        "n_augmented"
                                    ]
                                ),

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


                    if (
                        len(model_records)
                        != 20
                    ):

                        raise RuntimeError(

                            f"{run_id}: expected "
                            f"20 evaluations, found "
                            f"{len(model_records)}"
                        )


                    new_records.extend(
                        model_records
                    )


                    trained_now += 1


                    # =========================================
                    # SAVE AFTER EVERY MODEL
                    # =========================================

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

                        "Completed this session:",

                        trained_now
                    )


                    print(

                        "Total records saved:",

                        len(combined)
                    )


    # ========================================================
    # FINAL CONFIRMATION INTEGRITY
    # ========================================================

    final = pd.read_csv(
        RESULTS_FILE
    )


    print()

    print(
        "=" * 76
    )

    print(
        "PHASE 5 CONFIRMATION "
        "FINAL INTEGRITY CHECK"
    )

    print(
        "=" * 76
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
    # Exactly 20 environments/model
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


    # Expected:
    #
    # EXP1:
    # 1500 × 4 = 6000
    #
    # EXP1B:
    # 1500 × 3 = 4500
    #
    # EXP1C:
    # 1500 × 13 = 19500
    #
    # TOTAL:
    # 30000


    assert len(final) == 30000


    assert (

        final[
            "run_id"
        ].nunique()

        == 1500
    )


    assert duplicates == 0


    assert (
        len(incomplete)
        == 0
    )


    assert (

        experiment_counts[
            "EXP1"
        ]

        == 6000
    )


    assert (

        experiment_counts[
            "EXP1B"
        ]

        == 4500
    )


    assert (

        experiment_counts[
            "EXP1C"
        ]

        == 19500
    )


    # ========================================================
    # Factor integrity
    # ========================================================

    assert set(

        final[
            "dataset_id"
        ].unique()

    ) == set(
        DATASETS
    )


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
    # D × A × M factorial integrity
    #
    # Each cell:
    #
    # 10 seeds × 20 environments
    #
    # = 200 rows
    # ========================================================

    dam_counts = (

        final

        .groupby([
            "dataset_id",
            "augmentation_id",
            "model_config",
        ])

        .size()
    )


    assert (
        dam_counts == 200
    ).all()


    print()

    print(
        "D x A x M cells:",
        len(dam_counts)
    )


    print(

        "Rows per D x A x M cell:",

        sorted(
            dam_counts.unique()
        )
    )


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

            ].astype(int)
        )


        assert observed == {
            expected_params
        }


    # ========================================================
    # Augmentation integrity
    # ========================================================

    a0_n_aug = set(

        final.loc[

            final[
                "augmentation_id"
            ]
            == "A0",

            "n_augmented",

        ].astype(int)
    )


    assert a0_n_aug == {
        0
    }


    non_a0_n_aug = set(

        final.loc[

            final[
                "augmentation_id"
            ]
            != "A0",

            "n_augmented",

        ].astype(int)
    )


    assert non_a0_n_aug == {
        500
    }


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

            ].astype(int)
        )


        assert observed_aug_seeds == {

            AUGMENTATION_SEED_OFFSET
            + seed
        }


    # ========================================================
    # Training size
    # ========================================================

    assert set(

        final[
            "n_train"
        ].astype(int)

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

            final[column]
            .notna()
            .all()
        )


        assert np.isfinite(

            final[column]

        ).all()


    # ========================================================
    # EXP1C shift-family integrity
    # ========================================================

    exp1c = (

        final[

            final[
                "experiment"
            ]
            == "EXP1C"

        ].copy()
    )


    temperature = (

        exp1c[

            exp1c[
                "shift_type"
            ]
            == "temperature"

        ]
    )


    humidity = (

        exp1c[

            exp1c[
                "shift_type"
            ]
            == "humidity"

        ]
    )


    print()

    print(

        "EXP1C temperature records:",

        len(temperature)
    )


    print(

        "EXP1C humidity records:",

        len(humidity)
    )


    assert (
        len(temperature)
        == 9000
    )


    assert (
        len(humidity)
        == 10500
    )


    # ========================================================
    # Temperature severity integrity
    # ========================================================

    assert set(

        temperature[
            "severity"
        ].astype(int)

    ) == {
        0,
        1,
        2,
        3,
        4,
        5,
    }


    assert set(

        temperature[
            "shift_value"
        ].astype(int)

    ) == {
        23,
        25,
        27,
        29,
        31,
        33,
    }


    # ========================================================
    # Humidity severity integrity
    # ========================================================

    assert set(

        humidity[
            "severity"
        ].astype(int)

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
        ].astype(int)

    ) == {
        65,
        70,
        75,
        80,
        85,
        90,
        92,
    }


    # ========================================================
    # Every D × A × M × seed must contain all 20 environments
    # ========================================================

    model_environment_counts = (

        final

        .groupby([
            "dataset_id",
            "augmentation_id",
            "model_config",
            "seed",
        ])

        .size()
    )


    assert (

        model_environment_counts
        == 20

    ).all()


    # ========================================================
    # Each D × A × M cell must contain all 10 seeds
    # ========================================================

    cell_seed_counts = (

        final

        .groupby([
            "dataset_id",
            "augmentation_id",
            "model_config",
        ])[
            "seed"
        ]

        .nunique()
    )


    assert (

        cell_seed_counts
        == 10

    ).all()


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


    # ========================================================
    # PASS
    # ========================================================

    print()

    print(
        "=" * 76
    )

    print(
        "PHASE 5 CONFIRMATION: PASS"
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()