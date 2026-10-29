"""TabM member: explicit fixed architecture (no dynamic activation dispatch).

Architecture copied explicitly from the Yandex TabM reference implementation
(https://github.com/yandex-research/tabm, ``tabm.py`` v0.0.3, Apache-2.0; vendored
at commit ``28e47ae301c92ec37787dde1ce923a0793f405b4``; license shipped as
``LICENSE.tabm``):

    backbone = 2 blocks of [LinearBatchEnsemble -> ReLU -> Dropout]
    head     = LinearEnsemble(width -> 1, k members)
    forward  = member outputs, averaged at inference

Only the components this model uses are reproduced.  Parameter layout, module
registration order, initialization order and random-number consumption mirror the
reference implementation, so training is bit-identical to it; the equivalence is
checked by ``src/qf3/s6_v1_tabm_equiv.py``.
"""

import torch
import torch.nn as nn


def _init_rsqrt_uniform_(tensor: torch.Tensor, d: int) -> torch.Tensor:
    """Uniform on [-1/sqrt(d), 1/sqrt(d)] (reference `_init_rsqrt_uniform_`)."""
    bound = d ** -0.5
    return nn.init.uniform_(tensor, -bound, bound)


class LinearBatchEnsemble(nn.Module):
    """BatchEnsemble linear layer: ``h = ((x * r) @ W.T) * s + b``."""

    def __init__(self, in_features: int, out_features: int, *, k: int,
                 first_scaling_random_signs: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(k, in_features))
        self.s = nn.Parameter(torch.empty(k, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self._first_scaling_random_signs = first_scaling_random_signs
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_rsqrt_uniform_(self.weight, self.in_features)
        if self._first_scaling_random_signs:
            with torch.inference_mode():
                self.r.bernoulli_(0.5).mul_(2).add_(-1)
        else:
            nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        bias_init = torch.empty(self.out_features, dtype=self.weight.dtype,
                                device=self.weight.device)
        bias_init = _init_rsqrt_uniform_(bias_init, self.in_features)
        with torch.inference_mode():
            self.bias.copy_(bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.r
        x = x @ self.weight.T
        x = x * self.s
        return x + self.bias


class LinearEnsemble(nn.Module):
    """``k`` parallel linear layers applied element-wise to a ``(B, k, D)`` input."""

    def __init__(self, in_features: int, out_features: int, *, k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(k, out_features))
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_rsqrt_uniform_(self.weight, self.in_features)
        _init_rsqrt_uniform_(self.bias, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1) @ self.weight
        return x.transpose(0, 1) + self.bias


class _BackboneBatchEnsemble(nn.Module):
    """``n_blocks`` blocks of [LinearBatchEnsemble -> ReLU -> Dropout]."""

    def __init__(self, d_in: int, n_blocks: int, d_block: int, dropout: float, k: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                LinearBatchEnsemble(d_in if index == 0 else d_block, d_block, k=k,
                                    first_scaling_random_signs=(index == 0)),
                nn.ReLU(),
                nn.Dropout(dropout))
            for index in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TabMWrapper(nn.Module):
    """k shared-parameter MLPs; inference averages the member predictions."""

    def __init__(self, d_in: int, k: int = 8, width: int = 128, blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.k = k
        self.backbone = _BackboneBatchEnsemble(d_in, blocks, width, dropout, k)
        self.head = LinearEnsemble(width, 1, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        repeated = x.unsqueeze(1).expand(-1, self.k, -1)
        return self.head(self.backbone(repeated)).squeeze(-1)
