import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Configurable MLP used in Phase 3.

    M1 = small
    M0 = Phase 1/2 baseline
    M2 = large
    """

    def __init__(self, model_id: str, input_dim: int = 8):
        super().__init__()

        architectures = {
            "M1": [16, 8],
            "M0": [32, 16],
            "M2": [64, 32, 16],
        }

        if model_id not in architectures:
            raise ValueError(
                f"Unknown model_id={model_id}. "
                f"Choose from {list(architectures.keys())}"
            )

        hidden_dims = architectures[model_id]

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

        self.model_id = model_id
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

    def forward(self, x):
        return self.network(x)


def build_phase3_model(model_id: str, input_dim: int = 8):
    return MLPRegressor(
        model_id=model_id,
        input_dim=input_dim,
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )