"""A low-rank adapter with an explicit multiplicative interaction branch.

For row-vector inputs x, this implements
    f(x) = B(Ax) + U((Px) * (Qx)).
The branch is deliberately small and transparent so its expressivity claims
can be tested independently of a Transformer implementation.
"""

import torch
from torch import nn


class MultiplicativeKoRA(nn.Module):
    """Linear LoRA residual plus rank-k pairwise interactions."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 interaction_rank: int = 4, alpha: float = 1.0,
                 interaction_alpha: float = 1.0):
        super().__init__()
        if min(in_features, out_features, rank, interaction_rank) <= 0:
            raise ValueError("all feature and rank values must be positive")
        self.alpha = alpha / rank
        self.interaction_alpha = interaction_alpha / interaction_rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.interaction_P = nn.Parameter(torch.empty(interaction_rank, in_features))
        self.interaction_Q = nn.Parameter(torch.empty(interaction_rank, in_features))
        self.interaction_U = nn.Parameter(torch.zeros(out_features, interaction_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.interaction_P, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.interaction_Q, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = (x @ self.lora_A.t()) @ self.lora_B.t()
        interaction = (x @ self.interaction_P.t()) * (x @ self.interaction_Q.t())
        return self.alpha * linear + self.interaction_alpha * (interaction @ self.interaction_U.t())


class SharedBilinearKoRA(nn.Module):
    """Parameter-efficient KoRA with interactions in shared LoRA coordinates.

    The interaction branch is U((P(Ax)) * (Q(Ax))).  It therefore pays for
    interaction projections in rank space rather than input space, making
    parameter-matched comparisons with LoRA meaningful.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8,
                 interaction_rank: int = 2, alpha: float = 1.0,
                 interaction_alpha: float = 1.0):
        super().__init__()
        if min(in_features, out_features, rank, interaction_rank) <= 0:
            raise ValueError("all feature and rank values must be positive")
        self.alpha = alpha / rank
        self.interaction_alpha = interaction_alpha / interaction_rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.interaction_P = nn.Parameter(torch.empty(interaction_rank, rank))
        self.interaction_Q = nn.Parameter(torch.empty(interaction_rank, rank))
        self.interaction_U = nn.Parameter(torch.zeros(out_features, interaction_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.interaction_P, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.interaction_Q, a=5 ** 0.5)

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.lora_A.t()
        linear = z @ self.lora_B.t()
        interaction = (z @ self.interaction_P.t()) * (z @ self.interaction_Q.t())
        return self.alpha * linear + self.interaction_alpha * (interaction @ self.interaction_U.t())
