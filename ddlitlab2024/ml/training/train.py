import argparse
from dataclasses import asdict
from functools import partial

from torch.profiler import profile, ProfilerActivity

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import wandb
import yaml
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddlitlab2024.dataset.pytorch import DDLITLab2024Dataset, Normalizer, worker_init_fn
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
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to the configuration file")
    parser.add_argument("--checkpoint", "-p", type=str, default=None, help="Path to the checkpoint to load")
    parser.add_argument(
        "--output", "-o", type=str, default="trajectory_transformer_model.pth", help="Path to save the model"
    )
    parser.add_argument("--decoder-pretraining", action="store_true", help="Unconditionally train the decoder first")
    parser.add_argument("--pretrained-decoder", type=str, default=None, help="Path to the pretrained decoder model")
    args = parser.parse_args()

    assert (
        args.config is not None or args.checkpoint is not None
    ), "Either a configuration file or a checkpoint must be provided"

    # Load the hyperparameters from the checkpoint
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint '{args.checkpoint}'")
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        params = checkpoint["hyperparams"]

    # Load the hyperparameters from the configuration file
    if args.config is not None:
        logger.info(f"Loading configuration file '{args.config}'")
        with open(args.config) as file:
            config_params = yaml.safe_load(file)

        if args.checkpoint is not None:
            logger.warning(
                "Both a configuration file and a checkpoint are provided. "
                "The configuration file will be used for the hyperparameters."
            )
            # Print the differences between the checkpoint and the configuration file
            for key, value in config_params.items():
                if key not in params:
                    logger.warning(f"Key '{key}' is not present in the checkpoint")
                elif value != params[key]:
                    logger.warning(f"Key '{key}' has a different value in the checkpoint: {params[key]} != {value}")

        # Now we are ready to use the configuration file
        params = config_params

    # Initialize the weights and biases logging
    run = wandb.init(entity="bitbots", project="ddlitlab-2024", config=params)

    # Load the dataset
    logger.info("Create dataset objects")
    dataset = DDLITLab2024Dataset(
        num_joints=params["num_joints"],
        num_frames_video=params["image_context_length"],
        num_samples_joint_trajectory_future=params["trajectory_prediction_length"],
        num_samples_joint_trajectory=params["action_context_length"],
        num_samples_imu=params["imu_context_length"],
        num_samples_joint_states=params["joint_state_context_length"],
        imu_representation=IMUEncoder.OrientationEmbeddingMethod(params["imu_orientation_embedding_method"]),
        use_action_history=params["use_action_history"],
        use_imu=params["use_imu"],
        use_joint_states=params["use_joint_states"],
        use_images=params["use_images"],
        use_game_state=params["use_gamestate"],
        image_resolution=params.get(
            "image_resolution", 480
        ),  # This parameter has been added later so we need to check if it is present
    )
    num_workers = 32 if not args.decoder_pretraining else 24
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

    # Get some samples to estimate the mean and std
    logger.info("Estimating normalization parameters")
    random_indices = np.random.randint(0, len(dataset), (params["num_normalization_samples"],))
    normalization_samples = torch.cat([dataset[i].joint_command for i in tqdm(random_indices)], dim=0)
    normalizer = Normalizer.fit(normalization_samples.to(device))

    # Initialize the Transformer model and optimizer, and move model to device
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
        image_use_final_avgpool=params.get("image_use_final_avgpool", True),
        image_resolution=params.get("image_resolution", 480),
        num_decoder_layers=params["num_decoder_layers"],
        trajectory_prediction_length=params["trajectory_prediction_length"],
        use_gamestate=params["use_gamestate"],
        encoder_patch_size=params["encoder_patch_size"],
    ).to(device)

    # Add normalization parameters to the model
    model.mean = normalizer.mean
    model.std = normalizer.std
    logger.info(f"Normalization values:\nJoint mean: {normalizer.mean}\nJoint std: {normalizer.std}")
    assert all(model.std != 0), "Normalization std is zero, this makes no sense. Some joints are constant."

    # Log gradients and parameters to wandb
    run.watch(model)

    # Load the model if a checkpoint is provided
    if args.checkpoint is not None:
        logger.info("Loading model from checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])

    # Load the pretrained decoder model if provided
    if args.pretrained_decoder is not None:
        logger.info("Loading pretrained decoder model")
        decoder_checkpoint = torch.load(args.pretrained_decoder, weights_only=True)
        model.load_state_dict(decoder_checkpoint["model_state_dict"], strict=False)

    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"])

    # Load the optimizer state if a checkpoint is provided
    if args.checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            logger.info("Loading optimizer state from checkpoint")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("No optimizer state found in the checkpoint")

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=params["lr"], total_steps=params["epochs"] * len(dataloader)
    )

    # Load the learning rate scheduler state if a checkpoint is provided
    if args.checkpoint is not None and False:
        if "lr_scheduler_state_dict" in checkpoint:
            logger.info("Loading learning rate scheduler state from checkpoint")
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        else:
            logger.warning("No learning rate scheduler state found in the checkpoint")

    # Create diffusion noise scheduler
    scheduler = DDIMScheduler(beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.config["num_train_timesteps"] = params["train_denoising_timesteps"]

    # Training loop
    for epoch in range(params["epochs"]):
        # Iterate over the dataset
        for i, batch in enumerate(pbar := tqdm(dataloader)):
            # Move the data to the device
            batch = {k: v.to(device, non_blocking=True) for k, v in asdict(batch).items() if v is not None}

            # Extract the target actions
            joint_targets = batch["joint_command"]

            # Extract the batch size of the current batch
            # It might be different from the batch size in the hyperparameters
            # due to the last batch being smaller
            bs = joint_targets.size(0)

            # Normalize the target actions
            joint_targets = normalizer.normalize(joint_targets)

            # Reset the gradients
            optimizer.zero_grad()

            # Sample a random timestep for each trajectory in the batch
            random_timesteps = (
                torch.randint(0, scheduler.config["num_train_timesteps"], (joint_targets.size(0),)).long().to(device)
            )

            # Sample gaussian noise to add to the entire trajectory
            noise = torch.randn_like(joint_targets).to(device)

            # Forward diffusion: Add noise to the entire trajectory at the random timestep
            noisy_trajectory = scheduler.add_noise(joint_targets, noise, random_timesteps)

            # Predict the error using the model
            if args.decoder_pretraining:
                predicted_traj = model.forward_with_context(
                    [torch.randn((bs, 10, params["hidden_dim"]), device=device)], noisy_trajectory, random_timesteps
                )
            else:
                predicted_traj = model(batch, noisy_trajectory, random_timesteps)

            # Compute the loss
            loss = F.mse_loss(predicted_traj, noise)

            if i % 20 == 0:
                pbar.set_postfix_str(
                    f"Epoch {epoch}, Loss: {loss.item():.05f}, LR: {lr_scheduler.get_last_lr()[0]:0.7f}"
                )
                run.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=(i + epoch * len(dataloader)))

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        # Save the model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "hyperparams": params,
            "current_epoch": epoch,
        }
        torch.save(checkpoint, args.output)

    # Finish the run cleanly
    run.finish()
