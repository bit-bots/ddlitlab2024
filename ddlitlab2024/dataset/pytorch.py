#!/usr/bin/env python
import os
import sqlite3
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from profilehooks import profile
from torch.utils.data import DataLoader, Dataset

from ddlitlab2024.dataset.models import JointState, RobotState
from ddlitlab2024.ml.model.encoder.imu import IMUEncoder
from ddlitlab2024.utils.utils import quats_to_5d


class DDLITLab2024Dataset(Dataset):
    @dataclass
    class Result:
        joint_command: torch.Tensor
        joint_command_history: torch.Tensor
        joint_state: torch.Tensor
        rotation: torch.Tensor
        game_state: torch.Tensor
        image_data: torch.Tensor
        image_stamps: torch.Tensor

        def shapes(self) -> dict[str, tuple[int, ...]]:
            return {k: v.shape for k, v in asdict(self).items()}

    def __init__(
        self,
        data_base_path: str,
        num_samples_imu: int = 100,
        imu_representation: IMUEncoder.OrientationEmbeddingMethod = IMUEncoder.OrientationEmbeddingMethod.QUATERNION,
        num_samples_joint_states: int = 100,
        num_samples_joint_trajectory: int = 100,
        num_samples_joint_trajectory_future: int = 10,
        sampling_rate: int = 100,
        max_fps_video: int = 10,
        num_frames_video: int = 50,
        trajectory_stride: int = 10,
    ):
        # Store the parameters
        self.num_samples_imu = num_samples_imu
        self.imu_representation = imu_representation
        self.num_samples_joint_states = num_samples_joint_states
        self.num_samples_joint_trajectory = num_samples_joint_trajectory
        self.num_samples_joint_trajectory_future = num_samples_joint_trajectory_future
        self.sampling_rate = sampling_rate
        self.max_fps_video = max_fps_video
        self.num_frames_video = num_frames_video
        self.trajectory_stride = trajectory_stride

        # The Data exists in a sqlite database
        assert data_base_path.endswith(".sqlite3"), "The database should be a sqlite file"
        assert os.path.exists(data_base_path), "The database file does not exist"
        self.data_base_path = data_base_path

        # Load the data from the database
        self.db_connection = sqlite3.connect(self.data_base_path)

        # Lock the database to prevent writing
        self.db_connection.execute("PRAGMA locking_mode = EXCLUSIVE")

        # Get the total length of the dataset in seconds
        cursor = self.db_connection.cursor()

        # SQL query that get the first and last timestamp of the joint command for each recording
        cursor.execute(
            "SELECT recording_id, COUNT(*) AS num_entries_in_recording FROM JointCommand GROUP BY recording_id"
        )
        recording_timestamps = cursor.fetchall()

        # Calculate how many batches can be build from each recording
        self.num_samples = 0
        self.sample_boundaries = []
        for recording_id, num_data_points in recording_timestamps:
            assert num_data_points > 0, "Recording length is negative or zero"
            total_samples_before = self.num_samples
            # Calculate the number of batches that can be build from the recording including the stride
            self.num_samples += int(
                (num_data_points - self.num_samples_joint_trajectory_future) / self.trajectory_stride
            )
            # Store the boundaries of the samples for later retrieval
            self.sample_boundaries.append((total_samples_before, self.num_samples, recording_id))

    def __len__(self):
        return self.num_samples

    def query_joint_data(
        self, recording_id: int, start_sample: int, num_samples: int, table: Literal["JointCommand", "JointState"]
    ) -> torch.Tensor:
        # Get the joint state
        raw_joint_data = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE recording_id = {recording_id} "
            f"ORDER BY stamp ASC LIMIT {num_samples} OFFSET {start_sample}",
            # TODO other direction  TODO make params correct
            self.db_connection,
        )

        # Convert to numpy array, keep only the joint angle columns in alphabetical order
        raw_joint_data = raw_joint_data[
            [
                JointState.head_pan.name,
                JointState.head_tilt.name,
                JointState.l_ankle_pitch.name,
                JointState.l_ankle_roll.name,
                JointState.l_elbow.name,
                JointState.l_hip_pitch.name,
                JointState.l_hip_roll.name,
                JointState.l_hip_yaw.name,
                JointState.l_knee.name,
                JointState.l_shoulder_pitch.name,
                JointState.l_shoulder_roll.name,
                JointState.r_ankle_pitch.name,
                JointState.r_ankle_roll.name,
                JointState.r_elbow.name,
                JointState.r_hip_pitch.name,
                JointState.r_hip_roll.name,
                JointState.r_hip_yaw.name,
                JointState.r_knee.name,
                JointState.r_shoulder_pitch.name,
                JointState.r_shoulder_roll.name,
            ]
        ].to_numpy(dtype=np.float32)

        # We don't need padding here, because we sample the data in the correct length for the targets
        return torch.from_numpy(raw_joint_data)

    def query_joint_data_history(
        self, recording_id: int, end_sample: int, num_samples: int, table: Literal["JointCommand", "JointState"]
    ) -> torch.Tensor:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples_to_query = end_sample - start_sample

        # Get the joint data
        raw_joint_data = self.query_joint_data(recording_id, start_sample, num_samples_to_query, table)

        # Pad the data if necessary, for the input data / history it might be necessary
        # during the startup / first samples
        # Zero pad the joint state if the number of samples is less than the required number of samples
        if raw_joint_data.shape[0] < num_samples:
            raw_joint_data = torch.cat(
                (
                    torch.zeros(
                        (num_samples - raw_joint_data.shape[0], raw_joint_data.shape[1]), dtype=raw_joint_data.dtype
                    ),
                    raw_joint_data,
                ),
                dim=0,
            )
            assert raw_joint_data.shape[0] == num_samples, "The padded array is not the correct shape"
            assert raw_joint_data[0, 0] == 0.0, "The array is not zero padded"

        return raw_joint_data

    def query_image_data(
        self, recording_id: int, end_sample: int, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples_to_query = end_sample - start_sample

        # Get the image data
        cursor = self.db_connection.cursor()
        cursor.execute(
            "SELECT stamp, data FROM Image WHERE recording_id = $1 ORDER BY stamp ASC LIMIT $2 OFFSET $3",
            (recording_id, num_samples_to_query, start_sample),
        )

        stamps = []
        image_data = []

        # Get the raw image data
        for stamp, data in cursor:
            # Deserialize the image data
            image_data.append(np.frombuffer(data, dtype=np.uint8).reshape(480, 480, 3))
            stamps.append(stamp)

        # Apply zero padding if necessary
        if len(image_data) < num_samples:
            image_data = [
                np.zeros((480, 480, 3), dtype=np.uint8) for _ in range(num_samples - len(image_data))
            ] + image_data
            stamps = [0.0 for _ in range(num_samples - len(stamps))] + stamps

        # Convert to tensor
        image_data = torch.from_numpy(np.stack(image_data, axis=0))
        stamps = torch.tensor(stamps)

        return stamps, image_data

    def query_imu_data(self, recording_id: int, end_sample: int, num_samples: int) -> torch.Tensor:
        # Handle lower bound
        start_sample = max(0, end_sample - num_samples)
        num_samples_to_query = end_sample - start_sample

        # Get the imu data
        raw_imu_data = pd.read_sql_query(
            f"SELECT * FROM Rotation WHERE recording_id = {recording_id} "
            f"ORDER BY stamp ASC LIMIT {num_samples_to_query} OFFSET {start_sample}",
            # TODO make params correct
            self.db_connection,
        )

        # Convert to numpy array
        raw_imu_data = raw_imu_data[["x", "y", "z", "w"]].to_numpy(dtype=np.float32)

        # Add padding if necessary (identity quaternion)
        if raw_imu_data.shape[0] < num_samples:
            identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            raw_imu_data = np.concatenate(
                (
                    np.tile(identity_quaternion, (num_samples - raw_imu_data.shape[0], 1)),
                    raw_imu_data,
                ),
                axis=0,
            )

            assert raw_imu_data.shape[0] == num_samples, "The padded array is not the correct shape"
            assert np.allclose(
                raw_imu_data[0], identity_quaternion
            ), "The array does not start with the identity quaternion, even though it is padded"

        # Convert to correct representation
        match self.imu_representation:
            case IMUEncoder.OrientationEmbeddingMethod.FIVE_DIM:
                imu_data = quats_to_5d(raw_imu_data)

            case IMUEncoder.OrientationEmbeddingMethod.QUATERNION:
                imu_data = raw_imu_data

            case rep:
                raise NotImplementedError(f"Unknown IMU representation {rep}")

        return torch.from_numpy(imu_data)

    def query_current_game_state(self, recording_id: int, stamp: float) -> torch.Tensor:
        cursor = self.db_connection.cursor()
        # Select last game state before the current stamp
        cursor.execute(
            "SELECT state FROM GameState WHERE recording_id = $1 AND stamp <= $2 ORDER BY stamp DESC LIMIT 1",
            (recording_id, stamp),
        )

        # Get the game state
        game_state: RobotState = cursor.fetchone()

        # If no game state is found set it to unknown
        if game_state is None:
            game_state = RobotState.UNKNOWN
        else:
            game_state = RobotState(game_state[0])

        return torch.tensor(int(game_state))

    @profile
    def __getitem__(self, idx: int) -> Result:
        # Find the recording that contains the sample
        for start_sample, end_sample, recording_id in self.sample_boundaries:
            if idx >= start_sample and idx < end_sample:
                boundary = (recording_id, start_sample)
                break
        assert boundary is not None, "Could not find the recording that contains the sample"
        recording_id, start_sample = boundary

        # We assume that joint command, imu and joint state have the sampling rate and are synchronized
        # Game state and image data are not synchronized with the other data

        # Calculate the sample index in the recording
        sample_index = int(idx - start_sample)
        # Calculate the time stamp of the sample
        stamp = sample_index / self.sampling_rate

        # Get the image data
        image_stamps, image_data = self.query_image_data(recording_id, sample_index, self.num_frames_video)

        # Get the joint command target (future)
        joint_command = self.query_joint_data(
            recording_id, sample_index, self.num_samples_joint_trajectory_future, "JointCommand"
        )

        # Get the joint command history
        joint_command_history = self.query_joint_data_history(
            recording_id, sample_index, self.num_samples_joint_trajectory, "JointCommand"
        )

        # Get the joint state
        joint_state = self.query_joint_data_history(
            recording_id, sample_index, self.num_samples_joint_states, "JointState"
        )

        # Get the robot rotation (IMU data)
        robot_rotation = self.query_imu_data(recording_id, sample_index, self.num_samples_imu)

        # Get the game state
        game_state = self.query_current_game_state(recording_id, stamp)

        return self.Result(
            joint_command=joint_command,
            joint_command_history=joint_command_history,
            joint_state=joint_state,
            image_data=image_data,
            image_stamps=image_stamps,
            rotation=robot_rotation,
            game_state=game_state,
        )

    @staticmethod
    def collate_fn(batch: Iterable[Result]) -> Result:
        return DDLITLab2024Dataset.Result(
            joint_command=torch.stack([x.joint_command for x in batch]),
            joint_command_history=torch.stack([x.joint_command_history for x in batch]),
            joint_state=torch.stack([x.joint_state for x in batch]),
            image_data=torch.stack([x.image_data for x in batch]),
            image_stamps=torch.stack([x.image_stamps for x in batch]),
            rotation=torch.stack([x.rotation for x in batch]),
            game_state=torch.tensor([x.game_state for x in batch]),
        )


# Some dummy code to test the dataset
if __name__ == "__main__":
    dataset = DDLITLab2024Dataset(os.path.join(os.path.dirname(__file__), "db.sqlite3"))

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=DDLITLab2024Dataset.collate_fn)

    # Plot the first sample
    import time

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    time1 = time.time()

    batch_0: DDLITLab2024Dataset.Result = next(dataloader.__iter__())

    time2 = time.time()
    print(time2 - time1)

    print(batch_0.shapes())

    for i in range(batch_0.joint_command.shape[0]):
        print(f"\n\n---- Element {i} ----\n")

        # Print the game state
        print(f"Gamestate: {RobotState.values()[batch_0.game_state[i]]}")

        # Plot the image data history as grid 10x5
        fig = plt.figure(figsize=(10, 10))
        plt.title("Image context")
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(5, 10),  # creates 2x2 grid of Axes
            axes_pad=0.1,  # pad between Axes in inch.
        )
        for ax, im in zip(grid, batch_0.image_data[i]):
            ax.imshow(im.numpy())
        plt.show()

        # Plot the joint command history and future (a subplot for each joint)
        plt.figure(figsize=(10, 10))
        plt.title("Joint states, commands history and future commands")
        for j in range(20):
            plt.subplot(5, 4, j + 1)
            plt.title(f"Joint {j}")

            # Plot the joint command history
            plt.plot(
                np.arange(-batch_0.joint_command_history.shape[1], 0),
                batch_0.joint_command_history[i, :, j].numpy(),
                label="Command history",
            )

            # Draw the future joint commands
            plt.plot(batch_0.joint_command[i, :, j].numpy(), label="Command future")

            # Draw the joint state history
            plt.plot(
                np.arange(-batch_0.joint_state.shape[1], 0),
                batch_0.joint_state[i, :, j].numpy(),
                label="Joint state",
            )
        plt.legend()
        plt.show()

        # Plot the rotation data
        plt.title("Rotation")
        plt.plot(batch_0.rotation[i].numpy())
        plt.legend(["x", "y", "z", "w"])
        plt.show()
