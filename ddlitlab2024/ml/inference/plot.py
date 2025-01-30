import argparse
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
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

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Inference Plot")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint to load")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps (not used for distilled)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    args = parser.parse_args()

    # Load the hyperparameters from the checkpoint
    logger.info(f"Loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    params = checkpoint["hyperparams"]

    logger.info("Load model")
    model = End2EndDiffusionTransformer(
        num_joints=params["num_joints"],
        hidden_dim=params["hidden_dim"],
        use_action_history=params["use_action_history"],
        num_action_history_encoder_layers=params["num_action_history_encoder_layers"],
        max_action_context_length=params["action_context_length"],
        use_imu=params["use_imu"],
        imu_orientation_embedding_method=IMUEncoder.OrientationEmbeddingMethod(
            params["imu_orientation_embedding_method"]
        ),
        num_imu_encoder_layers=params["num_imu_encoder_layers"],
        imu_context_length=params["imu_context_length"],
        use_joint_states=params["use_joint_states"],
        joint_state_encoder_layers=params["joint_state_encoder_layers"],
        joint_state_context_length=params["joint_state_context_length"],
        use_images=params["use_images"],
        image_sequence_encoder_type=SequenceEncoderType(params["image_sequence_encoder_type"]),
        image_encoder_type=ImageEncoderType(params["image_encoder_type"]),
        num_image_sequence_encoder_layers=params["num_image_sequence_encoder_layers"],
        image_context_length=params["image_context_length"],
        num_decoder_layers=params["num_decoder_layers"],
        trajectory_prediction_length=params["trajectory_prediction_length"],
    ).to(device)
    normalizer = Normalizer(model.mean, model.std)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(normalizer.mean)

    # Create diffusion noise scheduler
    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config["num_train_timesteps"] = params["train_denoising_timesteps"]
    scheduler.set_timesteps(args.steps)

    # Create Dataset object
    dataset = DDLITLab2024Dataset(
        num_joints=params["num_joints"],
        num_frames_video=params["image_context_length"],
        num_samples_joint_trajectory_future=params["trajectory_prediction_length"],
        num_samples_joint_trajectory=params["action_context_length"],
        num_samples_imu=params["imu_context_length"],
        num_samples_joint_states=params["joint_state_context_length"],
    )

    # Create DataLoader object
    num_workers = 5
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=DDLITLab2024Dataset.collate_fn,
        persistent_workers=num_workers > 1,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    dataloader = iter(dataloader)

    for _ in range(args.num_samples):
        batch = next(dataloader)
        # Move the data to the device
        batch = {k: v.to(device) for k, v in asdict(batch).items()}

        # Extract the target actions
        joint_targets = batch["joint_command"]

        # Sample our initial trajectory to denoise
        noisy_trajectory = torch.randn_like(joint_targets).to(device)
        trajectory = noisy_trajectory

        if params.get("distilled_decoder", False):
            # Directly predict the trajectory based on the noise
            with torch.no_grad():
                trajectory = model(batch, noisy_trajectory, torch.tensor([0], device=device))
        else:
            # Perform the denoising process
            scheduler.set_timesteps(args.steps)
            for t in scheduler.timesteps:
                with torch.no_grad():
                    # Predict the noise residual
                    noise_pred = model(batch, trajectory, torch.tensor([t], device=device))

                    # Update the trajectory based on the predicted noise and the current step of the denoising process
                    trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

        # Undo the normalization
        trajectory = normalizer.denormalize(trajectory)
        noisy_trajectory = normalizer.denormalize(noisy_trajectory)

        # Plot the trajectory context, the noisy trajectory, the denoised trajectory
        # and the target trajectory for each joint
        plt.figure(figsize=(10, 10))
        for j in range(params["num_joints"]):
            plt.subplot(5, 4, j + 1)

            joint_command_context = batch["joint_command_history"][0, :, j].cpu().numpy()
            plt.plot(np.arange(len(joint_command_context)), joint_command_context, label="Context")
            plt.plot(
                np.arange(
                    len(joint_command_context), len(joint_command_context) + params["trajectory_prediction_length"]
                ),
                noisy_trajectory[0, :, j].cpu().numpy(),
                label="Noisy Trajectory",
            )
            plt.plot(
                np.arange(
                    len(joint_command_context), len(joint_command_context) + params["trajectory_prediction_length"]
                ),
                joint_targets[0, :, j].cpu().numpy(),
                label="Target Trajectory",
            )
            plt.plot(
                np.arange(
                    len(joint_command_context), len(joint_command_context) + params["trajectory_prediction_length"]
                ),
                trajectory[0, :, j].cpu().numpy(),
                label="Denoised Trajectory",
            )
            plt.title(f"Joint {dataset.joint_names[j]}")
        plt.legend()
        plt.show()
        plt.close()
