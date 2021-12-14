#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

Past = Tuple[torch.Tensor, torch.Tensor]


class BaseAttention(nn.Module):
    """
    Tensor          Type            Shape
    ==========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, dropout: float = 0.1, max_position_embeddings: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # register buffer for masked_bias and max_position_embeddings
        # copied from Huggingface's implementation (see causal_masking routine)
        self.mpe = max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((self.mpe, self.mpe), dtype=torch.uint8)).view(
                1, 1, self.mpe, self.mpe
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

    def causal_masking(self, x, q, k):
        """
        This routine is based off (and nearly identical to) the code in lines
        #188-#193 of Huggingface's transformers/models/gpt2/modeling_gpt2.py.
        (Version 4.7.0)
        """
        q_len, k_len = q.size(-2), k.size(-2)
        causal_mask = self.bias[:, :, k_len - q_len : k_len, :k_len].bool()
        return torch.where(causal_mask, x, self.masked_bias.to(x.dtype))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.matmul(q, k.transpose(-1, -2))
        x /= math.sqrt(k.size(-1))
        x = self.causal_masking(x, q, k)
        x = nn.Softmax(dim=-1)(x)
        x = self.dropout(x)
        x = torch.matmul(x, v)
        return x


class MultiHeadAttention(BaseAttention):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    mask            bool            (..., query_len, kv_len)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dropout: float = 0.1):
        super().__init__(dropout)
        self.heads = heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Split the tensors to multi-heads.
        q = q.view(q.size()[:-1] + (self.heads, q.size(-1) // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k.size(-1) // self.heads))
        v = v.view(v.size()[:-1] + (self.heads, v.size(-1) // self.heads))

        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)

        if mask is not None:
            mask = mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one.
        return (
            super()
            .forward(q, k, v, mask)
            .transpose(-3, -2)
            .contiguous()
            .view(q.size()[:-3] + (q.size(-2), v.size(-1) * self.heads))
        )


class AttentionLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    q               float           (..., query_len, dims)
    k               float           (..., kv_len, dims)
    v               float           (..., kv_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., query_len, past_len + kv_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., query_len, dims)
    output 2 (*)    float           (..., past_len + kv_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dims: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past: Optional[Past] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Past]:
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.attn(q, k, v, mask)
        x = self.linear(x)
        return x, (k, v)
