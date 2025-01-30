import argparse
from dataclasses import asdict
from functools import partial

import torch
import torch.nn.functional as F  # noqa
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddlitlab2024.dataset.pytorch import DDLITLab2024Dataset, worker_init_fn
from ddlitlab2024.ml import logger
from ddlitlab2024.ml.model import End2EndDiffusionTransformer
from ddlitlab2024.ml.model.encoder.image import ImageEncoderType, SequenceEncoderType
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix tqdm when the terminal width changes, this is for some reason not a default, therfore we make it one
tqdm = partial(tqdm, dynamic_ncols=True)

if __name__ == "__main__":
    logger.info("Starting training")
    logger.info(f"Using device {device}")
    # TODO wandb

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Distills the multi-step diffusion model into a single-step model")
    parser.add_argument("config", type=str, help="Path to the training configuration file")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint to load for the teacher model")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="distilled_trajectory_transformer_model.pth",
        help="Path to save the distilled model",
    )
    args = parser.parse_args()

    # Load the hyperparameters from the checkpoint
    logger.info(f"Loading checkpoint '{args.checkpoint}' for the teacher model and as a base for the student model")
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    teacher_params = checkpoint["hyperparams"]

    # Load the hyperparameters from the configuration file
    logger.info(f"Loading configuration file '{args.config}'")
    with open(args.config) as file:
        params = yaml.safe_load(file)

    # Print the differences between the checkpoint and the configuration file
    for key, value in params.items():
        if key not in teacher_params:
            logger.warning(f"Parameter '{key}' in the config not found in the checkpoints hyperparameters")
        elif value != teacher_params[key]:
            logger.warning(
                f"Parameter '{key}' has a different value in the teacher checkpoint: {teacher_params[key]} != {value}"
            )

    # Flag the student model as distilled
    params["distilled_decoder"] = True

    # Load the dataset (primary for example conditioning)
    logger.info("Create dataset objects")
    dataset = DDLITLab2024Dataset(
        num_joints=params["num_joints"],
        num_frames_video=params["image_context_length"],
        num_samples_joint_trajectory_future=params["trajectory_prediction_length"],
        num_samples_joint_trajectory=params["action_context_length"],
        num_samples_imu=params["imu_context_length"],
        num_samples_joint_states=params["joint_state_context_length"],
    )
    num_workers = 5
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=DDLITLab2024Dataset.collate_fn,
        persistent_workers=num_workers > 1,
        # prefetch_factor=10 * num_workers,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    model_config = dict(
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
    )

    # Initialize the Transformer model and optimizer, and move model to device
    teacher_model = End2EndDiffusionTransformer(**model_config).to(device)

    # Utilize an Exponential Moving Average (EMA) for the model to smooth out the training process
    teacher_ema = EMA(teacher_model, beta=0.999)

    # Load the model if a checkpoint is provided
    logger.info(f"Loading model from {checkpoint}")
    teacher_ema.load_state_dict(torch.load(checkpoint, weights_only=True))

    # Clone the model
    student_model = End2EndDiffusionTransformer(**model_config).to(device)

    # Load the same checkpoint into the student model
    # I load it from disk do avoid any potential issues when copying the model
    student_ema = EMA(student_model, beta=0.999)
    logger.info(f"Loading model from {checkpoint}")
    student_ema.load_state_dict(torch.load(checkpoint, weights_only=True))

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=params["lr"])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=params["lr"], total_steps=params["epochs"] * len(dataloader)
    )

    # Create diffusion noise scheduler
    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config["num_train_timesteps"] = params["train_denoising_timesteps"]

    # Training loop
    for epoch in range(params["epochs"]):
        mean_loss = 0

        # Iterate over the dataset
        for i, batch in enumerate(pbar := tqdm(dataloader)):
            # Move the data to the device
            batch = {k: v.to(device) for k, v in asdict(batch).items()}

            # Extract the target actions
            joint_targets = batch["joint_command"]

            # Sample our initial trajectory to denoise
            noisy_trajectory = torch.randn_like(joint_targets).to(device)

            # Reset the gradients
            optimizer.zero_grad()

            with torch.no_grad():
                # Clone the noisy trajectory because we want to keep the original as input for the student model
                trajectory = noisy_trajectory.clone()

                # Perform the embedding of the conditioning
                embedded_input = teacher_model.encode_input_data(batch)

                scheduler.set_timesteps(params["distill_teacher_inference_steps"])
                for t in scheduler.timesteps:
                    with torch.no_grad():
                        # Predict the noise residual
                        noise_pred = teacher_model.forward_with_context(
                            embedded_input, trajectory, torch.full((joint_targets.size(0),), t, device=device)
                        )

                        # Update the trajectory based on the predicted noise and
                        # the current step of the denoising process
                        trajectory = scheduler.step(noise_pred, t, trajectory).prev_sample

            # Predict the denoised trajectory directly using the student model
            # (null the timestep, as we are doing a single step prediction)
            student_trajectory_prediction = student_model.forward_with_context(
                embedded_input, noisy_trajectory, torch.zeros(joint_targets.size(0), device=device)
            )

            # Compute the loss
            loss = F.mse_loss(student_trajectory_prediction, trajectory)

            mean_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            student_ema.update()

            pbar.set_postfix_str(
                f"Epoch {epoch}, Loss: {mean_loss / (i + 1):.05f}, LR: {lr_scheduler.get_last_lr()[0]:0.7f}"
            )

        # Save the model
        checkpoint = {
            "model_state_dict": student_ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "hyperparams": params,
            "current_epoch": epoch,
        }
        torch.save(checkpoint, args.output)
