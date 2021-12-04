import math

import torch
import torch.nn as nn


class Activation(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """

    def __init__(self, activation="gelu"):
        super().__init__()
        if activation == "gelu":
            self.activation = self.gelu_new
        elif activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
            self.activation = lambda x: x * self.sigmoid(x)

    def gelu_new(self, x):
        """
        Implementation of the GELU activation function currently in Google BERT
        repo (identical to OpenAI GPT). Also see the Gaussian Error Linear
        Units paper: https://arxiv.org/abs/1606.08415
        Note: This routine is taken from Huggingface/transformers V4.7.0
        """
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class PositionwiseFeedForward(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """

    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(dims, dims * rate)
        self.act = Activation()
        self.proj = nn.Linear(dims * rate, dims)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.drop(x)
        return x
