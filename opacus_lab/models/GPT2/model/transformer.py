#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from opacus_lab.models.GPT2.model.attention import AttentionLayer, Past
from opacus_lab.models.GPT2.model.embedding import PositionalEmbedding, TokenEmbedding
from opacus_lab.models.GPT2.model.feedforward import PositionwiseFeedForward
from opacus_lab.models.GPT2.model.masking import FutureMasking, PadMasking


def factorize_linear_layer(LinearLayer, rank):
    U, S, Vh = torch.linalg.svd(LinearLayer.weight, full_matrices=False)
    V = Vh.mH
    lr_S = S[:rank]
    lr_U = U[:, 0:rank]
    lr_V = V.t()[:rank]
    out_features = lr_U.shape[0]
    in_features = lr_V.shape[1]
    bias = LinearLayer.bias is not None
    lr_LinearLayer = FactorizedLinear(in_features, out_features, rank, bias=bias)
    lr_LinearLayer.R.weight = nn.Parameter(torch.sqrt(lr_S).diag() @ lr_V)
    lr_LinearLayer.L.weight = nn.Parameter(lr_U @ torch.sqrt(lr_S).diag())
    if bias:
        lr_LinearLayer.L.bias = nn.Parameter(LinearLayer.bias)
    return lr_LinearLayer


def lrp_linear_layer(LinearLayer, rank):
    o, i = LinearLayer.weight.shape
    bias = LinearLayer.bias is not None
    lrp_LinearLayer = LowRankPerturbedLinear(i, o, rank, bias=bias)
    lrp_LinearLayer.core.weight = nn.Parameter(LinearLayer.weight, requires_grad=False)
    if bias:
        lrp_LinearLayer.core.bias = nn.Parameter(LinearLayer.bias, requires_grad=False)
    return lrp_LinearLayer


class FactorizedLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super().__init__()
        self.R = nn.Linear(in_features, rank, bias=False)
        self.L = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor):
        return self.L(self.R(x))


class LowRankPerturbedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scale: float = 0.0001,
        bias: bool = True,
    ):
        super().__init__()
        self.core = nn.Linear(in_features, out_features, bias=bias)
        self.LR = FactorizedLinear(in_features, out_features, rank, bias=bias)
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return self.scale * self.LR(x) + self.core(x)


class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(self, heads: int, dims: int, rate: int, dropout: float = 0.1):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = nn.LayerNorm(dims)
        self.ln_ff = nn.LayerNorm(dims)

    def forward(
        self,
        x: torch.Tensor,
        past: Optional[Past] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)
        x = x + a
        x = x + self.ff(self.ln_ff(x))
        return x


class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(
        self,
        layers: int,
        pad_idx: int,
        words: int,
        seq_len: int,
        heads: int,
        dims: int,
        rate: int = 4,
        dropout: float = 0.1,
        finetune: int = -1,
        lm_head_rank: int = 768,
        use_low_rank_head: bool = False,
        perturb: bool = True,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()
        self.positional_embedding = PositionalEmbedding(seq_len, dims)
        self.token_embedding = TokenEmbedding(words, dims)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList(
            [TransformerLayer(heads, dims, rate, dropout) for _ in range(layers)]
        )
        self.ln_head = nn.LayerNorm(dims)
        self.finetune = finetune

        if use_low_rank_head and lm_head_rank < dims:
            if perturb:
                self.lm_head = LowRankPerturbedLinear(
                    dims, words, rank=lm_head_rank, bias=False
                )
            else:
                self.lm_head = FactorizedLinear(
                    dims, words, rank=lm_head_rank, bias=False
                )
        else:
            self.lm_head = nn.Linear(dims, words, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        past: Optional[List[Past]] = None,
        use_grad_ckpt: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:

        torch.set_grad_enabled(self.finetune < 0)
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        for i, transformer in enumerate(self.transformers):
            torch.set_grad_enabled(self.finetune <= i)
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)

            x = transformer(x, past[i] if past is not None else None, mask)

        torch.set_grad_enabled(self.finetune <= i + 1)
        x = self.ln_head(x)
        x = self.lm_head(x)
        return x
