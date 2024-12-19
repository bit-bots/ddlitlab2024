from types import SimpleNamespace

import numpy as np
import pytest

joint_names = [
    "RShoulderPitch",
    "LShoulderPitch",
    "RShoulderRoll",
    "LShoulderRoll",
    "RElbow",
    "LElbow",
    "RHipYaw",
    "LHipYaw",
    "RHipRoll",
    "LHipRoll",
    "RHipPitch",
    "LHipPitch",
    "RKnee",
    "LKnee",
    "RAnklePitch",
    "LAnklePitch",
    "RAnkleRoll",
    "LAnkleRoll",
    "HeadPan",
    "HeadTilt",
]

positions = [
    -np.pi,
    -3.14159,
    -2.0,
    -1.0,
    -0.5,
    -0.1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.1,
    0.5,
    1.0,
    2.0,
    3.14159,
    np.pi,
]


@pytest.fixture
def imu_msg():
    return SimpleNamespace(x=1.0, y=2.0, z=3.0, w=4.0)


@pytest.fixture
def joint_command_msg():
    return SimpleNamespace(joint_names=joint_names, positions=positions)


@pytest.fixture
def joint_position_msg():
    return SimpleNamespace(name=joint_names, position=positions)
