import torch
from torch import nn

CORE_LOSS_KEY = "core_loss"


class DummyLoss(nn.Module):
    """Dummy loss for compatibility."""

    def forward(self, *args, **kwargs):
        return {CORE_LOSS_KEY: torch.tensor(0.0)}
