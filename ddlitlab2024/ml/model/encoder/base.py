import torch
from torch import nn

from ddlitlab2024.ml.model.misc import PositionalEncoding


class BaseEncoder(nn.Module):
    """
    Transformer encoder that encodes a sequence of input vectors into context tokens.
    """

    def __init__(
        self, input_dim: int, patch_size: int, hidden_dim: int, num_layers: int, num_heads: int, max_seq_len: int
    ):
        """
        Initializes the module.

        :param input_dim: The number of input dimensions.
        :param patch_size: The size of the patches for the convolutional embedding.
        :param hidden_dim: The number of hidden dimensions.
        :param num_layers: The number of transformer layers.
        :param num_heads: The number of attention heads.
        :param max_seq_len: The maximum length of the input sequences (used for positional encoding
        """
        super().__init__()
        # Embed into non-overlapping patches
        self.embedding = nn.Conv1d(input_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input vectors into context tokens.

        :param past_actions: The input vectors. Shape: (batch_size, seq_len, input_dim)
        :return: The encoded context tokens. Shape: (batch_size, seq_len, hidden_dim)
        """
        # Embed the input
        x = self.embedding(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Positional encoding
        x = self.positional_encoding(x)
        # Pass through the transformer encoder
        return self.transformer_encoder(x)
