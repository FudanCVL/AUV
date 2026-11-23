import torch.nn as nn
from typing import List, Dict
import torch


class LossFnWrapper(nn.Module):
    """
    Wrapper that calls all loss functions in a list and combines their outputs.
    """

    def __init__(self, loss_fns: List[nn.Module]):
        super().__init__()
        self.loss_fns = nn.ModuleList(loss_fns)

    def forward(self, outs_batch, targets_batch) -> Dict[str, torch.Tensor]:
        all_losses = {}
        for loss_fn in self.loss_fns:
            losses = loss_fn(outs_batch, targets_batch)
            for k, v in losses.items():
                if k in all_losses:
                    all_losses[k] += v
                else:
                    all_losses[k] = v
        return all_losses
