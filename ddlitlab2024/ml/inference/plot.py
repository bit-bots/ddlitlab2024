from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from ddlitlab2024.dataset.pytorch import DDLITLab2024Dataset, Normalizer, worker_init_fn
from ddlitlab2024.ml import logger
from ddlitlab2024.ml.model import End2EndDiffusionTransformer
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    logger.info("Starting")
    logger.info(f"Using device {device}")

    hidden_dim = 256
    num_layers = 4
    num_heads = 4
    action_context_length = 100
    trajectory_prediction_length = 10
    batch_size = 1
    lr = 1e-4
    train_denoising_timesteps = 1000
    image_context_length = 10
    action_context_length = 100
    imu_context_length = 100
    joint_state_context_length = 100
    num_normalization_samples = 50
    num_joints = 20
    checkpoint = "/homes/17vahl/ddlitlab2024/ddlitlab2024/ml/training/trajectory_transformer_model_fixed_norm.pth"

    logger.info("Load model")
    model = End2EndDiffusionTransformer(
        num_joints=20,
        hidden_dim=hidden_dim,
        use_action_history=True,
        num_action_history_encoder_layers=2,
        max_action_context_length=action_context_length,
        use_imu=True,
        imu_orientation_embedding_method=IMUEncoder.OrientationEmbeddingMethod.QUATERNION,
        num_imu_encoder_layers=2,
        max_imu_context_length=imu_context_length,
        use_joint_states=True,
        joint_state_encoder_layers=2,
        max_joint_state_context_length=joint_state_context_length,
        use_images=True,
        image_sequence_encoder_type=SequenceEncoderType.TRANSFORMER,
        image_encoder_type=ImageEncoderType.RESNET18,
        num_image_sequence_encoder_layers=1,
        max_image_context_length=image_context_length,
        num_decoder_layers=4,
        trajectory_prediction_length=trajectory_prediction_length,
    ).to(device)
    normalizer = Normalizer(model.mean, model.std)
    model = EMA(model)
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    model.eval()
    print(normalizer.mean)

    num_samples = 10
    inference_denosing_timesteps = 30

    # Create diffusion noise scheduler
    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config["num_train_timesteps"] = train_denoising_timesteps
    scheduler.set_timesteps(inference_denosing_timesteps)

    # Create Dataset object
    dataset = DDLITLab2024Dataset(
        num_joints=num_joints,
        num_frames_video=image_context_length,
        num_samples_joint_trajectory_future=trajectory_prediction_length,
        num_samples_joint_trajectory=action_context_length,
        num_samples_imu=imu_context_length,
        num_samples_joint_states=joint_state_context_length,
    )

    # Create DataLoader object
    num_workers = 5
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DDLITLab2024Dataset.collate_fn,
        persistent_workers=num_workers > 1,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    dataloader = iter(dataloader)

    for _ in range(num_samples):
        batch = next(dataloader)
        # Move the data to the device
        batch = {k: v.to(device) for k, v in asdict(batch).items()}

        # Extract the target actions
        joint_targets = batch["joint_command"]

        # Sample our initial trajectory to denoise
        noisy_trajectory = torch.randn_like(joint_targets).to(device)
        trajectory = noisy_trajectory

        # Perform the denoising process
        scheduler.set_timesteps(inference_denosing_timesteps)
        for t in scheduler.timesteps:
            with torch.no_grad():
                # Predict the noise residual
                noise_pred = model(batch, trajectory, torch.tensor([t], device=device))

                # Update the trajectory based on the predicted noise and the current step of the denoising process
                trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

        # Undo the normalization
        print(normalizer.mean)
        trajectory = normalizer.denormalize(trajectory)
        noisy_trajectory = normalizer.denormalize(noisy_trajectory)

        # Plot the trajectory context, the noisy trajectory, the denoised trajectory
        # and the target trajectory for each joint
        plt.figure(figsize=(10, 10))
        for j in range(num_joints):
            plt.subplot(5, 4, j + 1)

            joint_command_context = batch["joint_command_history"][0, :, j].cpu().numpy()
            plt.plot(np.arange(len(joint_command_context)), joint_command_context, label="Context")
            plt.plot(
                np.arange(len(joint_command_context), len(joint_command_context) + trajectory_prediction_length),
                noisy_trajectory[0, :, j].cpu().numpy(),
                label="Noisy Trajectory",
            )
            plt.plot(
                np.arange(len(joint_command_context), len(joint_command_context) + trajectory_prediction_length),
                joint_targets[0, :, j].cpu().numpy(),
                label="Target Trajectory",
            )
            plt.plot(
                np.arange(len(joint_command_context), len(joint_command_context) + trajectory_prediction_length),
                trajectory[0, :, j].cpu().numpy(),
                label="Denoised Trajectory",
            )
            plt.title(f"Joint {dataset.joint_names[j]}")
        plt.legend()
        plt.show()
        plt.close()
