import datetime
import math
import random

import cv2
import numpy as np
from sqlalchemy.orm import Session
from tqdm import tqdm

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import (
    GameState,
    Image,
    JointCommands,
    JointStates,
    Recording,
    RobotState,
    Rotation,
    TeamColor,
)


def insert_recordings(db_session: Session, n) -> list[int]:
    logger.debug("Inserting recordings...")
    for i in range(n):
        db_session.add(
            Recording(
                allow_public=True,
                original_file=f"dummy_original_file{i}",
                team_name=f"dummy_team_name{i}",
                team_color=random.choice(list(TeamColor)),
                robot_type=f"dummy_robot_type{i}",
                start_time=datetime.datetime.now(),
                location=f"dummy_location{i}",
                simulated=True,
                img_width_scaling=1.0,
                img_height_scaling=1.0,
            ),
        )
    recording = db_session.query(Recording).order_by(Recording._id.desc()).limit(n).all()
    if recording is None:
        raise ValueError("Failed to insert recordings")
    return [r._id for r in reversed(recording)]


def insert_images(db_session: Session, recording_ids: list[int], n: int, step: int) -> None:
    logger.info("Generating images...")

    def generate_test_image(width: int, height: int, timestamp: float) -> np.ndarray:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw a blue rectangle in the top left corner
        cv2.rectangle(img, (0, 0), (width // 2, height // 2), (255, 0, 0), -1)
        # Draw a red rectangle in the bottom right corner
        cv2.rectangle(img, (width // 2, height // 2), (width, height), (0, 0, 255), -1)
        # Write the string "RED" in red on the top right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "RED", (width - 100, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Write the string "BLUE" in blue on the top right corner below the red text
        cv2.putText(img, "BLUE", (width - 100, 100), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # Write the string "GREEN" in green on the top right corner below the blue text
        cv2.putText(img, "GREEN", (width - 100, 150), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Draw a white circle in the center
        cv2.circle(img, (width // 2, height // 2), 50, (255, 255, 255), -1)
        # Draw a smaller circle that changes color with the timestamp
        color = (int(255 * (1 + np.sin(timestamp)) / 2), int(255 * (1 + np.cos(timestamp)) / 2), 0)
        cv2.circle(img, (width // 2, height // 2), 25, color, -1)
        # Draw a text with the timestamp
        cv2.putText(img, f"{timestamp:.2f}", (10, height - 10), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for recording_id in tqdm(recording_ids):
        # Get width and height from the recording
        recording = db_session.query(Recording).get(recording_id)
        if recording is None:
            raise ValueError(f"Recording '{recording_id}' not found")
        for i in range(0, n, step):
            db_session.add(
                Image(
                    stamp=i / 100,
                    recording_id=recording_id,
                    image=generate_test_image(recording.img_height, recording.img_width, i / 100),
                )
            )


def insert_rotations(db_session: Session, recording_ids: list[int], n: int, speed=0.1) -> None:
    logger.info("Generating rotations...")
    for recording_id in tqdm(recording_ids):
        x_shift = random.random()
        y_shift = random.random()
        z_shift = random.random()
        w_shift = random.random()

        for i in range(n):
            db_session.add(
                Rotation(
                    stamp=i / 100,
                    recording_id=recording_id,
                    x=math.sin(i * speed + x_shift),
                    y=math.sin(i * speed + y_shift),
                    z=math.sin(i * speed + z_shift),
                    w=math.sin(i * speed + w_shift),
                ),
            )


def insert_joint_states(db_session: Session, recording_ids: list[int], n: int, speed: float = 0.2) -> None:
    logger.info("Generating joint states...")
    for recording_id in tqdm(recording_ids):
        offsets = [random.random() for _ in range(20)]
        for i in range(n):
            db_session.add(
                JointStates(
                    stamp=i / 100,
                    recording_id=recording_id,
                    r_shoulder_pitch=math.sin(speed * i + offsets[0]) + math.pi,
                    l_shoulder_pitch=math.sin(speed * i + offsets[1]) + math.pi,
                    r_shoulder_roll=math.sin(speed * i + offsets[2]) + math.pi,
                    l_shoulder_roll=math.sin(speed * i + offsets[3]) + math.pi,
                    r_elbow=math.sin(speed * i + offsets[4]) + math.pi,
                    l_elbow=math.sin(speed * i + offsets[5]) + math.pi,
                    r_hip_yaw=math.sin(speed * i + offsets[6]) + math.pi,
                    l_hip_yaw=math.sin(speed * i + offsets[7]) + math.pi,
                    r_hip_roll=math.sin(speed * i + offsets[8]) + math.pi,
                    l_hip_roll=math.sin(speed * i + offsets[9]) + math.pi,
                    r_hip_pitch=math.sin(speed * i + offsets[10]) + math.pi,
                    l_hip_pitch=math.sin(speed * i + offsets[11]) + math.pi,
                    r_knee=math.sin(speed * i + offsets[12]) + math.pi,
                    l_knee=math.sin(speed * i + offsets[13]) + math.pi,
                    r_ankle_pitch=math.sin(speed * i + offsets[14]) + math.pi,
                    l_ankle_pitch=math.sin(speed * i + offsets[15]) + math.pi,
                    r_ankle_roll=math.sin(speed * i + offsets[16]) + math.pi,
                    l_ankle_roll=math.sin(speed * i + offsets[17]) + math.pi,
                    head_pan=math.sin(speed * i + offsets[18]) + math.pi,
                    head_tilt=math.sin(speed * i + offsets[19]) + math.pi,
                ),
            )


def insert_joint_commands(db_session: Session, recording_ids: list[int], n: int, speed: float = 0.2) -> None:
    logger.info("Generating joint commands...")
    for recording_id in tqdm(recording_ids):
        offsets = [random.random() for _ in range(20)]
        for i in range(n):
            db_session.add(
                JointCommands(
                    stamp=i / 100,
                    recording_id=recording_id,
                    r_shoulder_pitch=math.sin(speed * i + offsets[0]) + math.pi,
                    l_shoulder_pitch=math.sin(speed * i + offsets[1]) + math.pi,
                    r_shoulder_roll=math.sin(speed * i + offsets[2]) + math.pi,
                    l_shoulder_roll=math.sin(speed * i + offsets[3]) + math.pi,
                    r_elbow=math.sin(speed * i + offsets[4]) + math.pi,
                    l_elbow=math.sin(speed * i + offsets[5]) + math.pi,
                    r_hip_yaw=math.sin(speed * i + offsets[6]) + math.pi,
                    l_hip_yaw=math.sin(speed * i + offsets[7]) + math.pi,
                    r_hip_roll=math.sin(speed * i + offsets[8]) + math.pi,
                    l_hip_roll=math.sin(speed * i + offsets[9]) + math.pi,
                    r_hip_pitch=math.sin(speed * i + offsets[10]) + math.pi,
                    l_hip_pitch=math.sin(speed * i + offsets[11]) + math.pi,
                    r_knee=math.sin(speed * i + offsets[12]) + math.pi,
                    l_knee=math.sin(speed * i + offsets[13]) + math.pi,
                    r_ankle_pitch=math.sin(speed * i + offsets[14]) + math.pi,
                    l_ankle_pitch=math.sin(speed * i + offsets[15]) + math.pi,
                    r_ankle_roll=math.sin(speed * i + offsets[16]) + math.pi,
                    l_ankle_roll=math.sin(speed * i + offsets[17]) + math.pi,
                    head_pan=math.sin(speed * i + offsets[18]) + math.pi,
                    head_tilt=math.sin(speed * i + offsets[19]) + math.pi,
                ),
            )


def insert_game_states(db_session: Session, recording_ids: list[int], n: int) -> None:
    logger.info("Generating game states...")
    for recording_id in tqdm(recording_ids):
        for i in range(n):
            db_session.add(
                GameState(
                    stamp=i / 100,
                    recording_id=recording_id,
                    state=random.choice(list(RobotState)),
                ),
            )


def insert_dummy_data(db_session: Session, num_recordings: int, num_samples_per_rec: int, image_step: int) -> None:
    logger.info("Inserting dummy data...")
    recording_ids: list[int] = insert_recordings(db_session, num_recordings)
    insert_images(db_session, recording_ids, num_samples_per_rec, image_step)
    insert_rotations(db_session, recording_ids, num_samples_per_rec)
    insert_joint_states(db_session, recording_ids, num_samples_per_rec)
    insert_joint_commands(db_session, recording_ids, num_samples_per_rec)
    insert_game_states(db_session, recording_ids, num_samples_per_rec)
    logger.info("Committing dummy data to database...")
    db_session.commit()
    logger.info(f"Dummy data inserted. Recording IDs: {recording_ids}")
