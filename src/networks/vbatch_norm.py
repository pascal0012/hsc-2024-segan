from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class VirtualBatchNorm(nn.Module):
    """
    Virtual batch normalization was implemented based on the implementation 
    from leftthomas on GitHub (https://github.com/leftthomas/SEGAN)
    """
    def __init__(self, num_features: int):
        super(VirtualBatchNorm, self).__init__()

        self.beta = nn.Parameter(torch.zeros((1, num_features, 1)))
        self.gamma = nn.Parameter(torch.ones((1, num_features, 1)))

    def forward(
        self,
        x: torch.Tensor,
        ref_mean: Optional[torch.Tensor] = None,
        ref_mean_sq: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        mean = x.mean(dim=[0, 2], keepdim=True)
        mean_sq = (x**2).mean(dim=[0, 2], keepdim=True)
    
        if ref_mean is None or ref_mean_sq is None:
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()

            return self.apply_norm(x, mean, mean_sq), mean, mean_sq
        else:
            new_factor = 1 / (1 + x.shape[0])
            mean = new_factor * mean + (1 - new_factor) * ref_mean
            mean_sq = new_factor * mean_sq + (1 - new_factor) * ref_mean_sq

            return self.apply_norm(x, ref_mean, ref_mean_sq)

    def apply_norm(
        self, x: torch.Tensor, mean: torch.Tensor, mean_sq: torch.Tensor
    ) -> torch.Tensor:
        std = torch.sqrt(1e-6 + mean_sq - mean**2)
        x = (x - mean) / std
        x = self.gamma * x + self.beta

        return x
