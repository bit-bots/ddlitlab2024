import torch
from torch import nn

from ddlitlab2024.ml.model.decoder import DiffusionActionGenerator
from ddlitlab2024.ml.model.encoder.game_state import GameStateEncoder
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType, image_sequence_encoder_factory
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder
from ddlitlab2024.ml.model.encoder.joint import JointEncoder
from ddlitlab2024.ml.model.misc import StepToken


class End2EndDiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_joints: int,
        hidden_dim: int,
        use_action_history: bool,
        num_action_history_encoder_layers: int,
        max_action_context_length: int,
        encoder_patch_size: int,
        use_imu: bool,
        imu_orientation_embedding_method: IMUEncoder.OrientationEmbeddingMethod,
        num_imu_encoder_layers: int,
        imu_context_length: int,
        use_joint_states: bool,
        joint_state_encoder_layers: int,
        joint_state_context_length: int,
        use_images: bool,
        image_encoder_type: ImageEncoderType,
        image_sequence_encoder_type: SequenceEncoderType,
        num_image_sequence_encoder_layers: int,
        image_context_length: int,
        image_use_final_avgpool: bool,
        image_resolution: int,
        use_gamestate: bool,
        num_decoder_layers: int,
        trajectory_prediction_length: int,
    ):
        super().__init__()

        # Define the network components

        # Step token used to encode the current step of the diffusion process
        self.step_encoding = StepToken(hidden_dim)

        # Action history encoder
        self.action_history_encoder = (
            JointEncoder(
                num_joints=num_joints,
                patch_size=encoder_patch_size,
                hidden_dim=hidden_dim,
                num_layers=num_action_history_encoder_layers,
                num_heads=4,
                max_seq_len=max_action_context_length,
            )
            if use_action_history
            else None
        )

        # IMU encoder
        self.imu_encoder = (
            IMUEncoder(
                orientation_embedding_method=imu_orientation_embedding_method,
                patch_size=encoder_patch_size,
                hidden_dim=hidden_dim,
                num_layers=num_imu_encoder_layers,
                num_heads=4,
                max_seq_len=imu_context_length,
            )
            if use_imu
            else None
        )

        # Joint states encoder
        self.joint_states_encoder = (
            JointEncoder(
                num_joints=num_joints,
                patch_size=encoder_patch_size,
                hidden_dim=hidden_dim,
                num_layers=joint_state_encoder_layers,
                num_heads=4,
                max_seq_len=joint_state_context_length,
            )
            if use_joint_states
            else None
        )

        # Image encoder
        self.image_sequence_encoder = (
            image_sequence_encoder_factory(
                encoder_type=image_sequence_encoder_type,
                image_encoder_type=image_encoder_type,
                hidden_dim=hidden_dim,
                num_layers=num_image_sequence_encoder_layers,
                max_seq_len=image_context_length,
                use_final_avgpool=image_use_final_avgpool,
                resolution=image_resolution,
            )
            if use_images
            else None
        )

        # Gamestate encoder
        self.game_state_encoder = GameStateEncoder(hidden_dim) if use_gamestate else None

        # Define the decoder model for the diffusion denoising process
        self.diffusion_action_generator = DiffusionActionGenerator(
            num_joints=num_joints,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=4,
            max_seq_len=trajectory_prediction_length,
        )

        # Store normalization parameters
        self.register_buffer("mean", torch.zeros(num_joints))
        self.register_buffer("std", torch.ones(num_joints))

    def encode_input_data(self, input_data: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Encodes the input data into context tokens. This function is split from the forward pass to allow for caching of
        the encoders during inference.

        :param input_data: The input data, containing information about the past actions, IMU data, joint states...
        :return: A list of tensors containing the encoded context for each input modality.
        """
        # Encode the context
        context = []

        # All of the following encoders can be cached during inference TODO
        if self.action_history_encoder is not None:
            context.append(self.action_history_encoder(input_data["joint_command_history"]))
        if self.imu_encoder is not None:
            context.append(self.imu_encoder(input_data["rotation"]))
        if self.joint_states_encoder is not None:
            context.append(self.joint_states_encoder(input_data["joint_state"]))
        if self.image_sequence_encoder is not None:
            context.append(self.image_sequence_encoder(input_data["image_data"]))
        if self.game_state_encoder is not None:
            context.append(self.game_state_encoder(input_data["game_state"]))

        # TODO utilize image time stamps

        return context

    def forward(
        self, input_data: dict[str, torch.Tensor], noisy_action_predictions: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        # Encode the context
        context = self.encode_input_data(input_data)

        # Run the forward pass
        return self.forward_with_context(context, noisy_action_predictions, step)

    def forward_with_context(
        self, context: list[torch.Tensor], noisy_action_predictions: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """
        This forward pass allows for the reuse of the context encoding. This can be useful when the context is
        precomputed, e.g., during inference.

        :param context: The precomputed context tokens.
        :param noisy_action_predictions: The noisy action predictions.
        :param step: The current step of the diffusion process.
        :return: The denoised action predictions.
        """

        # Generate step token to encode the current step of the diffusion process
        step_token = self.step_encoding(step)

        # Concatenate the context
        context_tensor = torch.cat(context + [step_token], dim=1)

        # Denoise the noisy action predictions
        return self.diffusion_action_generator(noisy_action_predictions, context_tensor)
