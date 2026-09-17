"""
Phase 6 Confirmation Experiment
================================

Independent replication/generalization using frozen G2.

Design
------
10 training-data compositions:
    D00-D09

5 augmentation strategies:
    A0-A4

3 model configurations:
    M0-M2

10 independent confirmation seeds:
    301, 312, 323, 334, 345,
    356, 367, 378, 389, 400

Total:
    10 x 5 x 3 x 10 = 1500 trained models

Each model is evaluated on 20 environments:

EXP1:
    normal
    warm
    humid
    cool

EXP1B:
    extreme_warm
    extreme_humid
    warm_humid

EXP1C:
    Temperature severity:
        23, 25, 27, 29, 31, 33

    Humidity severity:
        65, 70, 75, 80, 85, 90, 92

Total evaluation records:
    1500 x 20 = 30000

IMPORTANT
---------
This is the frozen Phase 6 confirmation experiment.

Do NOT:
- change G2 based on results
- change augmentations based on results
- change model configurations based on results
- remove difficult cells
- change seeds
- stop based on significance
- inspect partial results to redesign the experiment

The complete frozen factorial must be run.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from src.augmentation_phase6 import (
    apply_phase6_augmentation,
)

from src.train_phase3 import (
    train_phase3_model,
)

from src.evaluate import (
    evaluate_model,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "phase6"

TRAIN_DIR = DATA / "training"
VALIDATION_DIR = DATA / "validation"
TEST_DIR = DATA / "test"
OOD_DIR = DATA / "ood_test"
SEVERITY_DIR = DATA / "severity_test"

OUTPUT = (
    ROOT
    / "results"
    / "experiment_6"
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
# FROZEN PHASE 6 CONFIRMATION FACTORS
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

SEEDS = [
    301,
    312,
    323,
    334,
    345,
    356,
    367,
    378,
    389,
    400,
]

AUGMENTATION_SEED_OFFSET = 100000

MODEL_PARAMETERS = {
    "M1": 289,
    "M0": 833,
    "M2": 3201,
}


# ============================================================
# TARGET COMPATIBILITY
#
# G2 stores:
#
#     growth
#
# The frozen Phase 3/5 training/evaluation pipeline expects:
#
#     target_growth
#
# We create target_growth IN MEMORY ONLY.
# The frozen G2 CSV files are never modified.
# ============================================================

G2_TARGET = "growth"

LEGACY_TARGET = "target_growth"


def make_legacy_target_compatible(df):

    output = df.copy(
        deep=True
    )

    if G2_TARGET not in output.columns:

        raise ValueError(
            f"Phase 6 dataframe does not contain "
            f"'{G2_TARGET}'."
        )

    output[LEGACY_TARGET] = (
        output[
            G2_TARGET
        ].to_numpy(
            dtype=float
        )
    )

    return output


# ============================================================
# EVALUATION ENVIRONMENTS
# ============================================================

ENVIRONMENTS = []


# ============================================================
# EXP1
# ============================================================

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


# ============================================================
# EXP1B
# ============================================================

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


# ============================================================
# EXP1C — TEMPERATURE
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


# ============================================================
# EXP1C — HUMIDITY
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


assert len(ENVIRONMENTS) == 20


# ============================================================
# PRELOAD EVALUATION DATA
# ============================================================

evaluation_data = {}


for info in ENVIRONMENTS:

    path = info[
        "path"
    ]

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    if len(df) != 1000:

        raise RuntimeError(
            f"Unexpected evaluation size: "
            f"{path} -> {len(df)}"
        )

    if (
        df
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(
            f"Missing values in "
            f"{path}"
        )

    evaluation_data[
        info["environment"]
    ] = make_legacy_target_compatible(
        df
    )


# ============================================================
# VALIDATION DATA
# ============================================================

validation_path = (
    VALIDATION_DIR
    / "normal.csv"
)


if not validation_path.exists():

    raise FileNotFoundError(
        validation_path
    )


validation_g2 = pd.read_csv(
    validation_path
)


if len(validation_g2) != 1000:

    raise RuntimeError(
        "Phase 6 validation set "
        "does not contain 1000 rows."
    )


if (
    validation_g2
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "Missing values in Phase 6 "
        "validation data."
    )


validation_df = (
    make_legacy_target_compatible(
        validation_g2
    )
)


# ============================================================
# FROZEN CONFIGURATION
# ============================================================

CONFIG = {

    "phase":
        "PHASE6",

    "generator":
        "G2",

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
        0.50,

    "model_parameters":
        MODEL_PARAMETERS,

    "target_column_g2":
        G2_TARGET,

    "target_column_training_interface":
        LEGACY_TARGET,

    "target_interface_note":
        (
            "growth is copied to target_growth "
            "in memory only for compatibility "
            "with the frozen Phase 3/5 training "
            "and evaluation pipeline."
        ),

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
            "Phase6 G2 normal.csv",
    },

    "n_models":
        1500,

    "n_environments":
        20,

    "expected_records":
        30000,

    "purpose":
        (
            "Phase 6 independent G2 "
            "confirmation/generalization experiment."
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
# MAIN
# ============================================================

def main():

    print("=" * 76)

    print(
        "PHASE 6 G2 CONFIRMATION"
    )

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


    assert expected_models == 1500

    assert expected_records == 30000


    print(
        "Generator: G2"
    )

    print(
        "Datasets:",
        DATASETS,
    )

    print(
        "Augmentations:",
        AUGMENTATION_IDS,
    )

    print(
        "Model configurations:",
        MODEL_CONFIGS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    print(
        "Models:",
        expected_models,
    )

    print(
        "Environments/model:",
        len(ENVIRONMENTS),
    )

    print(
        "Expected records:",
        expected_records,
    )


    # ========================================================
    # SAVE CONFIGURATION
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
    # LOAD RESUME STATE
    # ========================================================

    existing = (
        load_existing()
    )

    completed = (
        completed_model_ids(
            existing
        )
    )


    print(
        "Completed models already present:",
        len(completed),
    )


    new_records = []

    trained_now = 0

    start = time.time()


    # ========================================================
    # DATASET LOOP
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


        original_train_df = (
            pd.read_csv(
                train_path
            )
        )


        if len(
            original_train_df
        ) != 1000:

            raise RuntimeError(
                f"{dataset_id} "
                "does not contain "
                "1000 rows."
            )


        if (
            original_train_df
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                f"{dataset_id} "
                "contains missing values."
            )


        if (
            G2_TARGET
            not in original_train_df.columns
        ):

            raise RuntimeError(
                f"{dataset_id} does not "
                f"contain '{G2_TARGET}'."
            )


        # ====================================================
        # SEED LOOP
        # ====================================================

        for seed in SEEDS:

            augmentation_seed = (

                AUGMENTATION_SEED_OFFSET

                + seed
            )


            # ================================================
            # AUGMENTATION LOOP
            # ================================================

            for augmentation_id in (
                AUGMENTATION_IDS
            ):


                # --------------------------------------------
                # IMPORTANT
                #
                # Augmentation is generated ONCE for
                # D x A x seed.
                #
                # The exact same augmented dataframe is
                # then shared by M0/M1/M2.
                # --------------------------------------------

                (
                    augmented_g2_df,
                    aug_metadata,

                ) = apply_phase6_augmentation(

                    df=
                        original_train_df,

                    augmentation_id=
                        augmentation_id,

                    seed=
                        augmentation_seed,

                    fraction=
                        0.50,
                )


                # ============================================
                # AUGMENTATION INTEGRITY
                # ============================================

                if len(
                    augmented_g2_df
                ) != 1000:

                    raise RuntimeError(
                        "Augmentation changed "
                        "training-set size: "
                        f"{dataset_id}, "
                        f"{augmentation_id}, "
                        f"seed={seed}"
                    )


                if (
                    augmented_g2_df
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

                    if augmentation_id == "A0"

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


                # ============================================
                # TARGET COMPATIBILITY COPY
                # ============================================

                augmented_train_df = (

                    make_legacy_target_compatible(
                        augmented_g2_df
                    )
                )


                # ============================================
                # MODEL LOOP
                # ============================================

                for model_config in (
                    MODEL_CONFIGS
                ):


                    run_id = (

                        f"G2_"

                        f"{dataset_id}_"

                        f"{augmentation_id}_"

                        f"{model_config}_"

                        f"seed{seed}"
                    )


                    # ========================================
                    # RESUME SUPPORT
                    # ========================================

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
                        run_id,
                    )


                    print(
                        f"Generator=G2, "
                        f"Dataset={dataset_id}, "
                        f"Augmentation="
                        f"{augmentation_id}, "
                        f"Model={model_config}, "
                        f"Seed={seed}, "
                        f"AugSeed="
                        f"{augmentation_seed}"
                    )


                    # ========================================
                    # TRAIN
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
                    )


                    if not np.isfinite(
                        val_loss
                    ):

                        raise RuntimeError(
                            "Non-finite validation "
                            f"loss: {run_id}"
                        )


                    # ========================================
                    # EVALUATE SAME MODEL
                    # ON ALL 20 ENVIRONMENTS
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


                        model_records.append({

                            "phase":
                                "PHASE6",

                            "generator":
                                "G2",

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
                                        "augmentation_fraction"
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


                    if len(
                        model_records
                    ) != 20:

                        raise RuntimeError(
                            f"{run_id} "
                            "did not produce "
                            "20 evaluations."
                        )


                    new_records.extend(
                        model_records
                    )

                    trained_now += 1


                    # ========================================
                    # SAVE AFTER EVERY MODEL
                    # ========================================

                    new_df = (
                        pd.DataFrame(
                            new_records
                        )
                    )


                    if existing.empty:

                        combined = (
                            new_df
                        )

                    else:

                        combined = (
                            pd.concat(
                                [
                                    existing,
                                    new_df,
                                ],
                                ignore_index=True,
                            )
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
                        trained_now,
                    )

                    print(
                        "Total records saved:",
                        len(combined),
                    )


    # ========================================================
    # FINAL CONFIRMATION INTEGRITY
    # ========================================================

    final = pd.read_csv(
        RESULTS_FILE
    )


    print()

    print("=" * 76)

    print(
        "PHASE 6 CONFIRMATION "
        "FINAL INTEGRITY CHECK"
    )

    print("=" * 76)


    print(
        "Records:",
        len(final),
    )

    print(
        "Unique models:",
        final[
            "run_id"
        ].nunique(),
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
        duplicates,
    )


    # ========================================================
    # 20 ENVIRONMENTS / MODEL
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
        len(incomplete),
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
    # GLOBAL SIZE CHECKS
    # ========================================================

    assert len(
        final
    ) == 30000


    assert (
        final[
            "run_id"
        ].nunique()
        == 1500
    )


    assert duplicates == 0


    assert len(
        incomplete
    ) == 0


    # ========================================================
    # EXPERIMENT SIZE CHECKS
    # ========================================================

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
    # GENERATOR CHECK
    # ========================================================

    assert set(
        final[
            "generator"
        ].unique()
    ) == {
        "G2"
    }


    # ========================================================
    # RUN TYPE
    # ========================================================

    assert set(
        final[
            "run_type"
        ].unique()
    ) == {
        "CONFIRMATION"
    }


    # ========================================================
    # DATASETS
    # ========================================================

    assert set(
        final[
            "dataset_id"
        ].unique()
    ) == set(
        DATASETS
    )


    # ========================================================
    # AUGMENTATIONS
    # ========================================================

    assert set(
        final[
            "augmentation_id"
        ].unique()
    ) == set(
        AUGMENTATION_IDS
    )


    # ========================================================
    # MODELS
    # ========================================================

    assert set(
        final[
            "model_config"
        ].unique()
    ) == set(
        MODEL_CONFIGS
    )


    # ========================================================
    # SEEDS
    # ========================================================

    assert set(
        final[
            "seed"
        ]
        .astype(int)
        .unique()
    ) == set(
        SEEDS
    )


    # ========================================================
    # D x A x M
    #
    # Each cell:
    #
    # 10 seeds x 20 environments
    # = 200 records
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


    assert len(
        dam_counts
    ) == 150


    assert (
        dam_counts
        == 200
    ).all()


    # ========================================================
    # D x A x M x SEED
    #
    # Every trained model:
    # 20 environments
    # ========================================================

    dams_counts = (

        final

        .groupby([
            "dataset_id",
            "augmentation_id",
            "model_config",
            "seed",
        ])

        .size()
    )


    assert len(
        dams_counts
    ) == 1500


    assert (
        dams_counts
        == 20
    ).all()


    # ========================================================
    # MODEL PARAMETER COUNTS
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
    # AUGMENTATION COUNTS
    # ========================================================

    a0_n_aug = set(

        final.loc[

            final[
                "augmentation_id"
            ]
            == "A0",

            "n_augmented",

        ]

        .astype(int)
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

        ]

        .astype(int)
    )


    assert non_a0_n_aug == {
        500
    }


    # ========================================================
    # TRAINING SET SIZE
    # ========================================================

    assert set(

        final[
            "n_train"
        ]

        .astype(int)

    ) == {
        1000
    }


    # ========================================================
    # AUGMENTATION SEED RULE
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


        assert observed_aug_seeds == {

            AUGMENTATION_SEED_OFFSET
            + seed

        }


    # ========================================================
    # NUMERICAL INTEGRITY
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


        values = (

            final[
                column
            ]

            .to_numpy(
                dtype=float
            )
        )


        assert np.isfinite(
            values
        ).all()


    assert (
        final[
            "mse"
        ] >= 0
    ).all()


    assert (
        final[
            "mae"
        ] >= 0
    ).all()


    # ========================================================
    # EXPECTED ENVIRONMENT NAMES
    # ========================================================

    expected_environment_names = {

        "normal",
        "warm",
        "humid",
        "cool",

        "extreme_warm",
        "extreme_humid",
        "warm_humid",

        "temp_s0",
        "temp_s1",
        "temp_s2",
        "temp_s3",
        "temp_s4",
        "temp_s5",

        "humidity_s0",
        "humidity_s1",
        "humidity_s2",
        "humidity_s3",
        "humidity_s4",
        "humidity_s5",
        "humidity_s6",
    }


    assert set(

        final[
            "environment"
        ].unique()

    ) == (
        expected_environment_names
    )


    # ========================================================
    # TEMPERATURE SEVERITY
    # ========================================================

    temp_rows = (

        final[

            final[
                "shift_type"
            ]
            == "temperature"

        ]
    )


    assert len(
        temp_rows
    ) == 9000


    assert set(

        temp_rows[
            "shift_value"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == set(
        TEMPERATURE_LEVELS
    )


    # ========================================================
    # HUMIDITY SEVERITY
    # ========================================================

    humidity_rows = (

        final[

            final[
                "shift_type"
            ]
            == "humidity"

        ]
    )


    assert len(
        humidity_rows
    ) == 10500


    assert set(

        humidity_rows[
            "shift_value"
        ]

        .dropna()

        .astype(int)

        .unique()

    ) == set(
        HUMIDITY_LEVELS
    )


    # ========================================================
    # EVERY MODEL MUST HAVE
    #
    # EXP1  = 4
    # EXP1B = 3
    # EXP1C = 13
    # ========================================================

    per_model_experiment = (

        final

        .groupby([
            "run_id",
            "experiment",
        ])

        .size()

        .unstack(
            fill_value=0
        )
    )


    assert (
        per_model_experiment[
            "EXP1"
        ]
        == 4
    ).all()


    assert (
        per_model_experiment[
            "EXP1B"
        ]
        == 3
    ).all()


    assert (
        per_model_experiment[
            "EXP1C"
        ]
        == 13
    ).all()


    # ========================================================
    # EVERY MODEL MUST HAVE
    #
    # 6 temperature severity environments
    # 7 humidity severity environments
    # ========================================================

    temp_per_model = (

        temp_rows

        .groupby(
            "run_id"
        )

        .size()
    )


    humidity_per_model = (

        humidity_rows

        .groupby(
            "run_id"
        )

        .size()
    )


    assert (
        temp_per_model
        == 6
    ).all()


    assert (
        humidity_per_model
        == 7
    ).all()


    # ========================================================
    # RUNTIME
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
    # FINAL PASS
    # ========================================================

    print()

    print("=" * 76)

    print(
        "PHASE 6 G2 CONFIRMATION: "
        "STRUCTURAL PASS"
    )

    print("=" * 76)

    print()

    print(
        "1500 models verified."
    )

    print(
        "30000 evaluation records verified."
    )

    print(
        "10 datasets verified."
    )

    print(
        "5 augmentations verified."
    )

    print(
        "3 model configurations verified."
    )

    print(
        "10 confirmation seeds verified."
    )

    print(
        "20 environments/model verified."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This establishes computational/"
        "structural integrity only."
    )

    print(
        "Scientific interpretation must be "
        "performed separately after the "
        "confirmation dataset is frozen."
    )


if __name__ == "__main__":

    main()
   