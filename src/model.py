import torch.nn as nn


class BaselineMLP(nn.Module):
    """
    Fixed baseline model M0.

    Architecture:
        8 -> 32 -> 16 -> 1
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)