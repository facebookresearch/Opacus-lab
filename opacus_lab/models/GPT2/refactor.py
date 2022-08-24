#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn

from opacus_lab.models.GPT2.model.attention import AttentionLayer
from opacus_lab.models.GPT2.model.embedding import PositionalEmbedding, TokenEmbedding
from opacus_lab.models.GPT2.model.feedforward import PositionwiseFeedForward
from opacus_lab.models.GPT2.model.transformer import Transformer, TransformerLayer


def refactor_transformer(
    GPT2,
    size="S",
    use_low_rank=False,
    lm_head_rank=768,
    dropout=0.0,
    perturb=False,
    vocab_size=50257,
):

    size_assertion_failure_str = f"Value {size} is not a valid size. "
    size_assertion_failure_str += 'Size must be one of: "S" (small), '
    size_assertion_failure_str += '"M" (medium), "L" (large),'
    size_assertion_failure_str += '"XL" (extra large),'
    size_assertion_failure_str += 'or "D" (distilled).'
    assert size in {"S", "D", "M", "L", "XL"}, size_assertion_failure_str

    size2dim = {"S": 768, "M": 1024, "L": 1280, "XL": 1600, "D": 768}
    size2blks = {"S": 12, "M": 24, "L": 36, "XL": 48, "D": 6}
    size2head = {"S": 12, "M": 16, "L": 20, "XL": 25, "D": 12}
    # specify some "architecture size" vars
    dim = size2dim[size]
    n_blks = size2blks[size]
    n_heads = size2head[size]

    # define a bunch of modular subroutines to complete the refactor

    def refactor_feedforward(GPT2MLP):
        FC = GPT2MLP.c_fc
        Proj = GPT2MLP.c_proj

        Feedforward = PositionwiseFeedForward(dim)
        Feedforward.fc.weight = nn.Parameter(FC.weight.t())
        Feedforward.fc.bias = nn.Parameter(FC.bias)
        Feedforward.proj.weight = nn.Parameter(Proj.weight.t())
        Feedforward.proj.bias = nn.Parameter(Proj.bias)
        return Feedforward

    def refactor_attention(GPT2Attention):
        Conv1D = GPT2Attention.c_attn
        Proj = GPT2Attention.c_proj

        Attention = AttentionLayer(n_heads, dim, 0.1)
        Attention.linear.weight = nn.Parameter(Proj.weight.t())
        Attention.linear.bias = nn.Parameter(Proj.bias)

        q_weight, k_weight, v_weight = torch.split(Conv1D.weight, [dim] * 3, dim=-1)
        q_bias, k_bias, v_bias = torch.split(Conv1D.bias, [dim] * 3, dim=-1)

        Attention.proj_q.weight = nn.Parameter(q_weight.t())
        Attention.proj_k.weight = nn.Parameter(k_weight.t())
        Attention.proj_v.weight = nn.Parameter(v_weight.t())

        Attention.proj_q.bias = nn.Parameter(q_bias)
        Attention.proj_k.bias = nn.Parameter(k_bias)
        Attention.proj_v.bias = nn.Parameter(v_bias)

        return Attention

    def refactor_block(GPT2Block):
        # 4X expansion rate is hardcoded below
        Block = TransformerLayer(n_heads, dim, 4)

        # first copy layernorm weights, no refactor needed
        Block.ln_attn.weight = nn.Parameter(GPT2Block.ln_1.weight)
        Block.ln_attn.bias = nn.Parameter(GPT2Block.ln_1.bias)
        Block.ln_ff.weight = nn.Parameter(GPT2Block.ln_2.weight)
        Block.ln_ff.bias = nn.Parameter(GPT2Block.ln_2.bias)

        # next refactor and copy the attention & FC layers
        Block.attn = refactor_attention(GPT2Block.attn)
        Block.ff = refactor_feedforward(GPT2Block.mlp)

        return Block

    def refactor_embeddings(GPT2):
        # num of pos emb hardcoded below
        wpe = PositionalEmbedding(1024, dim)
        wte = TokenEmbedding(vocab_size, dim)

        # If we extend the vocabulary, we only set the new embeddings
        gpt2_vocab_size = GPT2.transformer.wte.weight.shape[0]
        wte.emb.weight.data[:gpt2_vocab_size] = GPT2.transformer.wte.weight.data
        wpe.emb.weight = nn.Parameter(GPT2.transformer.wpe.weight)
        return wpe, wte

    def refactor_head(GPT2):
        head = nn.Linear(dim, vocab_size, bias=False)
        ln_head = nn.LayerNorm(dim)
        ln_head.weight = nn.Parameter(GPT2.transformer.ln_f.weight)
        ln_head.bias = nn.Parameter(GPT2.transformer.ln_f.bias)

        gpt2_vocab_size = GPT2.lm_head.weight.shape[0]
        head.weight.data[:gpt2_vocab_size] = GPT2.lm_head.weight.data

        return head, ln_head

    # a few hardcoded values:
    # pad index token id = 50256
    # max sequence len = 1024
    # head expansion factor = 4
    T = Transformer(
        n_blks,
        50256,
        vocab_size,
        1024,
        n_heads,
        dim,
        4,
        use_low_rank_head=use_low_rank,
        lm_head_rank=lm_head_rank,
        perturb=perturb,
        dropout=dropout,
    )

    # first refactor the transformer stack
    for i, block in enumerate(GPT2.transformer.h):
        T.transformers[i] = refactor_block(block)

    # then refactor the embeddings
    positional_emb, token_emb = refactor_embeddings(GPT2)
    T.positional_embedding = positional_emb
    T.token_embedding = token_emb

    # finally don't forget about head's layernorm & linear
    head, ln_head = refactor_head(GPT2)
    T.lm_head = head
    T.ln_head = ln_head

    return T


def test_refactor(pretrained, refactored, vocab_size=50257, exact=True):
    V = pretrained.transformer.wte.weight.shape[0]
    B, T, V = 2, 5, vocab_size
    string = torch.randint(V, size=(B, T), dtype=torch.int32)
    pretrained = pretrained.eval()
    refactored = refactored.eval()
    X = pretrained(string)
    Y = refactored(string)

    # If refactored model has larger vocabulary, we restrict it to the pretrained model's vocab
    if Y.shape[1] > X.logits.shape[1]:
        Y = Y[:, : X.logits.shape[1]]

    if exact:
        return Y.equal(X.logits)
    else:
        return torch.norm(X.logits - Y) / torch.norm(X.logits) < 1e-4
