from ddlitlab2024.ml.model.encoder.base import BaseEncoder


class JointEncoder(BaseEncoder):
    """
    Joint encoder that encodes the joint states of the robot.
    """

    def __init__(
        self, num_joints: int, patch_size: int, hidden_dim: int, num_layers: int, num_heads: int, max_seq_len: int
    ):
        """
        Initializes the module.

        :param num_joints: The number of joints in the robot.
        :param patch_size: The size of the patches for the convolutional embedding.
        :param hidden_dim: The number of hidden dimensions.
        :param num_layers: The number of transformer layers.
        :param num_heads: The number of attention heads.
        :param max_seq_len: The maximum length of the input sequences (used for positional encoding
        """
        super().__init__(
            input_dim=num_joints,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
        )
