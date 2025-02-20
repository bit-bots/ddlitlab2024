from enum import Enum

from ddlitlab2024.ml.model.encoder.base import BaseEncoder


class IMUEncoder(BaseEncoder):
    """
    Transformer encoder that encodes the action history of the robot.
    """

    class OrientationEmbeddingMethod(Enum):
        """
        Enum class for the orientation embedding methods.
        """

        QUATERNION = "quaternion"
        FIVE_DIM = "five_dim"  # Axis-angle with 2d vector for the angle

    def __init__(
        self,
        orientation_embedding_method: OrientationEmbeddingMethod,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int,
    ):
        """
        Initializes the module.

        :param orientation_embedding_method: The method used to embed the orientation data.
        :param patch_size: The size of the patches for the convolutional embedding.
        :param hidden_dim: The number of hidden dimensions.
        :param num_layers: The number of transformer layers.
        :param num_heads: The number of attention heads.
        :param max_seq_len: The maximum length of the input sequences (used for positional encoding
        """

        # Calculate the number of input features
        match orientation_embedding_method:
            case IMUEncoder.OrientationEmbeddingMethod.QUATERNION:
                input_features = 4
            case IMUEncoder.OrientationEmbeddingMethod.FIVE_DIM:
                input_features = 5

        super().__init__(
            input_dim=input_features,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
