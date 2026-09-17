import copy
import random

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.model_phase3 import build_phase3_model


FEATURES = [
    "air_temperature",
    "humidity",
    "water_temperature",
    "ph",
    "ec",
    "light",
    "co2",
    "growth_stage",
]


TARGET = "target_growth"


def set_seed(seed):
    """
    Set all random seeds used by the Phase 3
    training pipeline.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_phase3_model(
    train_df,
    validation_df,
    seed,
    model_id,
    epochs=150,
    batch_size=64,
    learning_rate=0.001,
):
    """
    Train one Phase 3 model.

    Everything is kept identical to the original
    Phase 1/2 training protocol except model_id.

    Model configurations:

        M1: 8 -> 16 -> 8 -> 1
        M0: 8 -> 32 -> 16 -> 1
        M2: 8 -> 64 -> 32 -> 16 -> 1
    """

    # --------------------------------------------------
    # Validate model ID
    # --------------------------------------------------

    valid_models = {
        "M0",
        "M1",
        "M2",
    }

    if model_id not in valid_models:
        raise ValueError(
            f"Invalid model_id '{model_id}'. "
            f"Expected one of {sorted(valid_models)}."
        )

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_seed(seed)

    # --------------------------------------------------
    # Prepare training data
    # --------------------------------------------------

    X_train = train_df[FEATURES].values.astype(
        np.float32
    )

    y_train = train_df[TARGET].values.astype(
        np.float32
    )

    # --------------------------------------------------
    # Prepare validation data
    # --------------------------------------------------

    X_val = validation_df[FEATURES].values.astype(
        np.float32
    )

    y_val = validation_df[TARGET].values.astype(
        np.float32
    )

    # --------------------------------------------------
    # Standardization
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on training data.
    # This is identical to Phase 1/2.
    # --------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    ).reshape(-1, 1)

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.float32
    ).reshape(-1, 1)

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    dataset = TensorDataset(
        X_train,
        y_train
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    # --------------------------------------------------
    # MODEL
    #
    # THIS is the experimental change in Phase 3.
    # --------------------------------------------------

    model = build_phase3_model(
        model_id=model_id
    )

    # --------------------------------------------------
    # Optimizer
    #
    # Same as Phase 1/2.
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # --------------------------------------------------
    # Loss
    #
    # Same as Phase 1/2.
    # --------------------------------------------------

    criterion = torch.nn.MSELoss()

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(epochs):

        model.train()

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            optimizer.step()

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        model.eval()

        with torch.no_grad():

            val_predictions = model(
                X_val
            )

            validation_loss = criterion(
                val_predictions,
                y_val
            ).item()

        # ----------------------------------------------
        # Keep best validation checkpoint
        # ----------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_model_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

    # --------------------------------------------------
    # Safety check
    # --------------------------------------------------

    if best_model_state is None:
        raise RuntimeError(
            "Training finished without "
            "a valid model checkpoint."
        )

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------

    model.load_state_dict(
        best_model_state
    )

    # --------------------------------------------------
    # Return same objects needed by evaluation
    # --------------------------------------------------

    return (
        model,
        scaler,
        best_validation_loss
    )