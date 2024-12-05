from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index, Integer, String, asc
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship
from sqlalchemy.types import LargeBinary

Base = declarative_base()

DEFAULT_IMG_SIZE = (480, 480)


class RobotState(str, Enum):
    POSITIONING = "POSITIONING"
    PLAYING = "PLAYING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls):
        return sorted([e.value for e in cls])

    def __int__(self):
        # Use index of sorted strings
        return self.values().index(self.value)


class TeamColor(str, Enum):
    BLUE = "BLUE"
    RED = "RED"
    YELLOW = "YELLOW"
    BLACK = "BLACK"
    WHITE = "WHITE"
    GREEN = "GREEN"
    ORANGE = "ORANGE"
    PURPLE = "PURPLE"
    BROWN = "BROWN"
    GRAY = "GRAY"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class Recording(Base):
    __tablename__ = "Recording"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    allow_public: Mapped[bool] = mapped_column(Boolean, default=False)
    original_file: Mapped[str] = mapped_column(String, nullable=False)
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    team_color: Mapped[Optional[TeamColor]] = mapped_column(String, nullable=True)
    robot_type: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    img_width: Mapped[int] = mapped_column(Integer, default=DEFAULT_IMG_SIZE[0])
    img_height: Mapped[int] = mapped_column(Integer, default=DEFAULT_IMG_SIZE[1])
    # Scaling factors for original image size to img_width x img_height
    img_width_scaling: Mapped[float] = mapped_column(Float, nullable=False)
    img_height_scaling: Mapped[float] = mapped_column(Float, nullable=False)

    images: Mapped[list["Image"]] = relationship("Image", back_populates="recording", cascade="all, delete-orphan")
    rotations: Mapped[list["Rotation"]] = relationship(
        "Rotation", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_states: Mapped[list["JointStates"]] = relationship(
        "JointStates", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_commands: Mapped[list["JointCommands"]] = relationship(
        "JointCommands", back_populates="recording", cascade="all, delete-orphan"
    )
    game_states: Mapped[list["GameState"]] = relationship(
        "GameState", back_populates="recording", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(img_width > 0),
        CheckConstraint(img_height > 0),
        CheckConstraint(team_color.in_(TeamColor.values())),
        CheckConstraint("end_time >= start_time"),
    )


class Image(Base):
    __tablename__ = "Image"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    # The image data should contain the image as bytes using an rgb8 format (3 channels) and uint8 type.
    # and should be of size (img_width, img_height) as specified in the recording (default 480x480)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="images")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        # Index to retrieve images in order from a given recording
        Index("idx_recording_stamp_image", "recording_id", asc("stamp")),
    )

    def __init__(
        self, stamp: float, image: np.ndarray, recording_id: int | None = None, recording: Recording | None = None
    ):
        assert image.dtype == np.uint8, "Image must be of type np.uint8"
        assert image.ndim == 3, "Image must have 3 dimensions"
        assert image.shape[2] == 3, "Image must have 3 channels"
        assert recording_id is not None or recording is not None, "Either recording_id or recording must be provided"

        if recording is None:
            super().__init__(stamp=stamp, recording_id=recording_id, data=image.tobytes())
        else:
            super().__init__(stamp=stamp, recording=recording, data=image.tobytes())


class Rotation(Base):
    __tablename__ = "Rotation"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)
    w: Mapped[float] = mapped_column(Float, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="rotations")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        CheckConstraint("x >= -1 AND x <= 1"),
        CheckConstraint("y >= -1 AND y <= 1"),
        CheckConstraint("z >= -1 AND z <= 1"),
        CheckConstraint("w >= -1 AND w <= 1"),
        # Index to retrieve rotations in order from a given recording
        Index("idx_recording_stamp_rotation", "recording_id", asc("stamp")),
    )


class JointStates(Base):
    __tablename__ = "JointStates"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LShoulderRoll")
    r_elbow: Mapped[float] = mapped_column(Float, default=0.0, name="RElbow")
    l_elbow: Mapped[float] = mapped_column(Float, default=0.0, name="LElbow")
    r_hip_yaw: Mapped[float] = mapped_column(Float, default=0.0, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, default=0.0, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, default=0.0, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, default=0.0, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, default=0.0, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, default=0.0, name="HeadTilt")

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_states")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        CheckConstraint("RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()"),
        CheckConstraint("LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()"),
        CheckConstraint("RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()"),
        CheckConstraint("LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()"),
        CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()"),
        CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()"),
        CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()"),
        CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()"),
        CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()"),
        CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()"),
        CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()"),
        CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()"),
        CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()"),
        CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()"),
        CheckConstraint("RAnklePitch >= 0 AND RAnklePitch < 2 * pi()"),
        CheckConstraint("LAnklePitch >= 0 AND LAnklePitch < 2 * pi()"),
        CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()"),
        CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()"),
        CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()"),
        CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()"),
        # Index to retrieve joint states in order from a given recording
        Index("idx_recording_stamp_joint_state", "recording_id", asc("stamp")),
    )


class JointCommands(Base):
    __tablename__ = "JointCommands"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LShoulderRoll")
    r_elbow: Mapped[float] = mapped_column(Float, default=0.0, name="RElbow")
    l_elbow: Mapped[float] = mapped_column(Float, default=0.0, name="LElbow")
    r_hip_yaw: Mapped[float] = mapped_column(Float, default=0.0, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, default=0.0, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, default=0.0, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, default=0.0, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, default=0.0, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, default=0.0, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, default=0.0, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, default=0.0, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, default=0.0, name="HeadTilt")

    recording: Mapped["Recording"] = relationship("Recording", back_populates="joint_commands")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        CheckConstraint("RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()"),
        CheckConstraint("LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()"),
        CheckConstraint("RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()"),
        CheckConstraint("LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()"),
        CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()"),
        CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()"),
        CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()"),
        CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()"),
        CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()"),
        CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()"),
        CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()"),
        CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()"),
        CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()"),
        CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()"),
        CheckConstraint("RAnklePitch >= 0 AND RAnklePitch < 2 * pi()"),
        CheckConstraint("LAnklePitch >= 0 AND LAnklePitch < 2 * pi()"),
        CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()"),
        CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()"),
        CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()"),
        CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()"),
        # Index to retrieve joint commands in order from a given recording
        Index("idx_recording_stamp_joint_command", "recording_id", asc("stamp")),
    )


class GameState(Base):
    __tablename__ = "GameState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    state: Mapped[RobotState] = mapped_column(String, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="game_states")

    __table_args__ = (
        CheckConstraint(state.in_(RobotState.values())),
        # Index to retrieve game states in order from a given recording
        Index("idx_recording_stamp_game_state", "recording_id", asc("stamp")),
    )


def stamp_to_seconds_nanoseconds(stamp: float) -> tuple[int, int]:
    seconds = int(stamp // 1)
    nanoseconds = int((stamp % 1) * 1e9)
    return seconds, nanoseconds


def stamp_to_nanoseconds(stamp: float) -> int:
    return int(stamp * 1e9)
