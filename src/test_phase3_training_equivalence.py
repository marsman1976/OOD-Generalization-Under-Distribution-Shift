import pandas as pd
import torch

from src.train import train_model
from src.train_phase3 import train_phase3_model
from src.evaluate import evaluate_model


TRAIN_PATH = "data/training/D00.csv"
VALIDATION_PATH = "data/validation/normal.csv"

SEED = 11


def main():

    print("=" * 70)
    print("PHASE 3 — TRAINING PIPELINE EQUIVALENCE AUDIT")
    print("=" * 70)

    # --------------------------------------------------
    # Load identical data
    # --------------------------------------------------

    print("\nLoading data...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Training rows:   {len(train_df)}"
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # --------------------------------------------------
    # OLD Phase 1/2 pipeline
    # --------------------------------------------------

    print("\nTraining OLD Phase 1/2 M0...")

    old_model, old_scaler, old_val_loss = (
        train_model(
            train_df=train_df,
            validation_df=validation_df,
            seed=SEED,
        )
    )

    # --------------------------------------------------
    # NEW Phase 3 pipeline using M0
    # --------------------------------------------------

    print("Training NEW Phase 3 M0...")

    new_model, new_scaler, new_val_loss = (
        train_phase3_model(
            train_df=train_df,
            validation_df=validation_df,
            seed=SEED,
            model_id="M0",
        )
    )

    # --------------------------------------------------
    # Validation loss comparison
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("VALIDATION LOSS")
    print("-" * 70)

    print(
        f"Old validation loss: "
        f"{old_val_loss:.12f}"
    )

    print(
        f"New validation loss: "
        f"{new_val_loss:.12f}"
    )

    val_difference = abs(
        old_val_loss - new_val_loss
    )

    print(
        f"Difference: "
        f"{val_difference:.12e}"
    )

    # --------------------------------------------------
    # Scaler comparison
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("SCALER")
    print("-" * 70)

    mean_difference = abs(
        old_scaler.mean_
        - new_scaler.mean_
    ).max()

    scale_difference = abs(
        old_scaler.scale_
        - new_scaler.scale_
    ).max()

    print(
        f"Max scaler mean difference: "
        f"{mean_difference:.12e}"
    )

    print(
        f"Max scaler scale difference: "
        f"{scale_difference:.12e}"
    )

    # --------------------------------------------------
    # Parameter comparison
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("TRAINED PARAMETERS")
    print("-" * 70)

    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    assert list(old_state.keys()) == list(
        new_state.keys()
    )

    max_parameter_difference = 0.0

    for key in old_state:

        difference = torch.max(
            torch.abs(
                old_state[key]
                - new_state[key]
            )
        ).item()

        max_parameter_difference = max(
            max_parameter_difference,
            difference,
        )

    print(
        f"Maximum parameter difference: "
        f"{max_parameter_difference:.12e}"
    )

    # --------------------------------------------------
    # Evaluation comparison
    # --------------------------------------------------

    old_metrics = evaluate_model(
        old_model,
        old_scaler,
        validation_df,
    )

    new_metrics = evaluate_model(
        new_model,
        new_scaler,
        validation_df,
    )

    print("\n" + "-" * 70)
    print("EVALUATION")
    print("-" * 70)

    print(
        f"Old MSE: {old_metrics['mse']:.12f}"
    )

    print(
        f"New MSE: {new_metrics['mse']:.12f}"
    )

    print(
        f"Old MAE: {old_metrics['mae']:.12f}"
    )

    print(
        f"New MAE: {new_metrics['mae']:.12f}"
    )

    mse_difference = abs(
        old_metrics["mse"]
        - new_metrics["mse"]
    )

    mae_difference = abs(
        old_metrics["mae"]
        - new_metrics["mae"]
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert val_difference < 1e-10, (
        "Validation losses differ."
    )

    assert mean_difference < 1e-12, (
        "Scaler means differ."
    )

    assert scale_difference < 1e-12, (
        "Scaler scales differ."
    )

    assert max_parameter_difference < 1e-10, (
        "Trained model parameters differ."
    )

    assert mse_difference < 1e-10, (
        "Evaluation MSE differs."
    )

    assert mae_difference < 1e-10, (
        "Evaluation MAE differs."
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PHASE 3 TRAINING PIPELINE "
        "EQUIVALENCE AUDIT: PASS"
    )
    print("=" * 70)

    print(
        "\nConclusion:\n"
        "The Phase 3 M0 training pipeline reproduces "
        "the original Phase 1/2 M0 training pipeline "
        "for the audited dataset and seed."
    )


if __name__ == "__main__":
    main()