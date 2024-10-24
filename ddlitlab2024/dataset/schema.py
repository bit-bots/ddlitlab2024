from datetime import datetime
from enum import Enum
from typing import List, Optional

from sqlalchemy import Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship, sessionmaker
from sqlalchemy.types import LargeBinary

from ddlitlab2024.dataset import logger

Base = declarative_base()


class RobotState(str, Enum):
    POSITIONING = "POSITIONING"
    PLAYING = "PLAYING"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


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
    location: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    img_width: Mapped[int] = mapped_column(Integer, default=480)
    img_height: Mapped[int] = mapped_column(Integer, default=480)
    img_width_scaling: Mapped[float] = mapped_column(Float, nullable=False)
    img_height_scaling: Mapped[float] = mapped_column(Float, nullable=False)

    images: Mapped[List["Image"]] = relationship("Image", back_populates="recording", cascade="all, delete-orphan")
    rotations: Mapped[List["Rotation"]] = relationship(
        "Rotation", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_states: Mapped[List["JointState"]] = relationship(
        "JointState", back_populates="recording", cascade="all, delete-orphan"
    )
    joint_commands: Mapped[List["JointCommand"]] = relationship(
        "JointCommand", back_populates="recording", cascade="all, delete-orphan"
    )
    game_states: Mapped[List["GameState"]] = relationship(
        "GameState", back_populates="recording", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(img_width > 0),
        CheckConstraint(img_height > 0),
        CheckConstraint(team_color.in_(TeamColor.values())),
    )


class Image(Base):
    __tablename__ = "Image"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="images")

    __table_args__ = (CheckConstraint("stamp >= 0"),)


class Rotation(Base):
    __tablename__ = "Rotation"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    roll: Mapped[float] = mapped_column(Float, nullable=False)
    pitch: Mapped[float] = mapped_column(Float, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="rotations")

    __table_args__ = (
        CheckConstraint("stamp >= 0"),
        CheckConstraint("roll >= 0 AND roll < 2 * pi()"),
        CheckConstraint("pitch >= 0 AND pitch < 2 * pi()"),
    )


class JointState(Base):
    __tablename__ = "JointState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LShoulderRoll")
    r_elbow: Mapped[float] = mapped_column(Float, nullable=False, name="RElbow")
    l_elbow: Mapped[float] = mapped_column(Float, nullable=False, name="LElbow")
    r_hip_yaw: Mapped[float] = mapped_column(Float, nullable=False, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, nullable=False, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, nullable=False, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, nullable=False, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, nullable=False, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, nullable=False, name="HeadTilt")

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
    )


class JointCommand(Base):
    __tablename__ = "JointCommand"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    r_shoulder_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RShoulderPitch")
    l_shoulder_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LShoulderPitch")
    r_shoulder_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RShoulderRoll")
    l_shoulder_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LShoulderRoll")
    r_elbow: Mapped[float] = mapped_column(Float, nullable=False, name="RElbow")
    l_elbow: Mapped[float] = mapped_column(Float, nullable=False, name="LElbow")
    r_hip_yaw: Mapped[float] = mapped_column(Float, nullable=False, name="RHipYaw")
    l_hip_yaw: Mapped[float] = mapped_column(Float, nullable=False, name="LHipYaw")
    r_hip_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RHipRoll")
    l_hip_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LHipRoll")
    r_hip_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RHipPitch")
    l_hip_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LHipPitch")
    r_knee: Mapped[float] = mapped_column(Float, nullable=False, name="RKnee")
    l_knee: Mapped[float] = mapped_column(Float, nullable=False, name="LKnee")
    r_ankle_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="RAnklePitch")
    l_ankle_pitch: Mapped[float] = mapped_column(Float, nullable=False, name="LAnklePitch")
    r_ankle_roll: Mapped[float] = mapped_column(Float, nullable=False, name="RAnkleRoll")
    l_ankle_roll: Mapped[float] = mapped_column(Float, nullable=False, name="LAnkleRoll")
    head_pan: Mapped[float] = mapped_column(Float, nullable=False, name="HeadPan")
    head_tilt: Mapped[float] = mapped_column(Float, nullable=False, name="HeadTilt")

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
    )


class GameState(Base):
    __tablename__ = "GameState"

    _id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stamp: Mapped[float] = mapped_column(Float, nullable=False)
    recording_id: Mapped[int] = mapped_column(Integer, ForeignKey("Recording._id"), nullable=False)
    state: Mapped[RobotState] = mapped_column(String, nullable=False)

    recording: Mapped["Recording"] = relationship("Recording", back_populates="game_states")

    __table_args__ = (CheckConstraint(state.in_(RobotState.values())),)


def main():
    logger.info("Creating database schema")
    engine = create_engine("sqlite:///data.sqlite")
    Base.metadata.create_all(engine)
    sessionmaker(bind=engine)()
    logger.info("Database schema created")


if __name__ == "__main__":
    main()
