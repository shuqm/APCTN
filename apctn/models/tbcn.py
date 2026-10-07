"""Transformer Brownian Covariance Network (TBCN).

This module implements the feature extractor described in the APCTN paper:
1) patch/token embedding of a 2-D CWT time-frequency image;
2) positional encoding + Transformer encoder (multi-head attention + MLP/GELU);
3) Brownian distance covariance (BDC) pooling across feature channels;
4) projection of the upper-triangular BDC matrix to the representation dimension
   used by prototypical contrastive learning and the adaptive classifier.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block compatible with older PyTorch versions."""

    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, C]
        y = self.norm1(x).transpose(0, 1)  # [N, B, C]
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y.transpose(0, 1))
        x = x + self.mlp(self.norm2(x))
        return x


class BrownianDistanceCovariancePool(nn.Module):
    """Compute a double-centered Euclidean-distance BDC feature matrix.

    The input is a set of local descriptors with shape [B, N, C].  Each channel
    is treated as a variable observed over N tokens. Pairwise Euclidean distances
    between channels are double-centered to obtain a Brownian-distance-covariance
    representation. The upper triangle is returned as a compact vector.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, return_matrix=False):
        if x.dim() != 3:
            raise ValueError("BDC input must have shape [batch, tokens, channels]")

        # [B, C, N]: C random variables, N observations per variable.
        x = x.transpose(1, 2)
        n_obs = x.size(-1)

        # Pairwise Euclidean distances between feature channels.
        xx = torch.sum(x * x, dim=-1, keepdim=True)
        dist2 = xx + xx.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
        dist = torch.sqrt(torch.clamp(dist2 / max(n_obs, 1), min=self.eps))

        # Double centering. The -1/2 factor gives a covariance-like BDC matrix.
        row_mean = dist.mean(dim=2, keepdim=True)
        col_mean = dist.mean(dim=1, keepdim=True)
        grand_mean = dist.mean(dim=(1, 2), keepdim=True)
        bdc = -0.5 * (dist - row_mean - col_mean + grand_mean)

        if return_matrix:
            return bdc

        c = bdc.size(1)
        tri = torch.triu_indices(c, c, device=bdc.device)
        return bdc[:, tri[0], tri[1]]


class TransformerBrownianCovarianceNetwork(nn.Module):
    """TBCN feature extractor for CWT time-frequency images."""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        out_dim=512,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.bdc_pool = BrownianDistanceCovariancePool()

        bdc_dim = embed_dim * (embed_dim + 1) // 2
        self.fc = nn.Linear(bdc_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, return_bdc=False):
        # CWT image -> token sequence.
        x = self.patch_embed(x)  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        if x.size(1) != self.pos_embed.size(1):
            # Interpolate positional encoding if a different input resolution is used.
            side_old = int(math.sqrt(self.pos_embed.size(1)))
            side_new = int(math.sqrt(x.size(1)))
            if side_new * side_new != x.size(1):
                raise ValueError("Token count must form a square grid for positional interpolation")
            pos = self.pos_embed.transpose(1, 2).reshape(1, self.embed_dim, side_old, side_old)
            pos = F.interpolate(pos, size=(side_new, side_new), mode="bicubic", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.pos_embed

        x = self.pos_drop(x + pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        bdc_matrix = self.bdc_pool(x, return_matrix=True)
        c = bdc_matrix.size(1)
        tri = torch.triu_indices(c, c, device=bdc_matrix.device)
        bdc_vector = bdc_matrix[:, tri[0], tri[1]]
        feat = self.out_norm(self.fc(bdc_vector))

        if return_bdc:
            return feat, bdc_matrix
        return feat
