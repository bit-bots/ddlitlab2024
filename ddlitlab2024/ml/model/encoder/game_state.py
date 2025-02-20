import torch
from torch import nn

from ddlitlab2024.dataset.models import RobotState


class GameStateEncoder(nn.Module):
    """
    Embeds the game state into learned context tokens.
    """

    def __init__(self, hidden_dim: int):
        """
        Initializes the module.
        """
        super().__init__()
        self.embedding = nn.Embedding(len(RobotState), hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input vectors into context tokens.

        :param x: The input states. Shape: (batch_size, num_game_states)
        :return: The encoded context tokens. Shape: (batch_size, hidden_dim)
        """
        # Embed the input
        return self.embedding(x).unsqueeze(1)
