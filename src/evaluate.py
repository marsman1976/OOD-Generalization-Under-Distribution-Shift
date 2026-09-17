import numpy as np
import torch

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


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


def evaluate_model(
    model,
    scaler,
    dataframe
):

    X = dataframe[FEATURES].values.astype(
        np.float32
    )

    y_true = dataframe[TARGET].values.astype(
        np.float32
    )

    X = scaler.transform(X)

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():

        y_pred = (
            model(X)
            .numpy()
            .flatten()
        )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    return {
        "mse": mse,
        "mae": mae
    }