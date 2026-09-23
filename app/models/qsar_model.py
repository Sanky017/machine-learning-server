"""
Model architecture definition.

Kept separate from train.py / inference.py so both import the exact same
class — otherwise torch.load() can silently mismatch layers.
"""

import torch.nn as nn


class QSARNet(nn.Module):
    """
    Minimal feedforward regressor over molecular descriptors.
    Replace/extend once your real QSAR target + feature set is finalized —
    input_dim must match len(featurize.FEATURE_NAMES).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
