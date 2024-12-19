from unittest.mock import Mock

import numpy as np
import pytest

from ddlitlab2024.dataset.converters.synced_data_converter import SyncedDataConverter
from ddlitlab2024.dataset.imports.data import InputData
from ddlitlab2024.dataset.models import Recording
from ddlitlab2024.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler
from ddlitlab2024.dataset.resampling.resampler import Sample


def test_all_synced_data_required(converter, recording, imu_msg, joint_position_msg, joint_command_msg):
    input = InputData()
    relative_timestamp = 1.13

    with pytest.raises(AssertionError):
        converter.convert_to_model(input, relative_timestamp, recording)

    input.joint_state = joint_position_msg
    with pytest.raises(AssertionError):
        converter.convert_to_model(input, relative_timestamp, recording)

    input.joint_command = joint_command_msg
    with pytest.raises(AssertionError):
        converter.convert_to_model(input, relative_timestamp, recording)

    input.rotation = imu_msg
    converter.convert_to_model(input, relative_timestamp, recording)


def test_resamples_synced_input_data(converter, input_data, recording):
    relative_timestamp = 1.13

    converter.convert_to_model(input_data, relative_timestamp, recording)

    converter.resampler.resample.assert_called_once_with(input_data, relative_timestamp)


def test_converts_all_resampled_rotations(converter, input_data, recording):
    converter.resampler.resample.return_value = [
        Sample(data=input_data, timestamp=0.0),
        Sample(data=input_data, timestamp=1.0),
    ]

    models = converter.convert_to_model(input_data, 1.13, recording)

    assert len(models.rotations) == 2
    assert models.rotations[0].stamp == 0.0
    assert models.rotations[1].stamp == 1.0

    for rotation in models.rotations:
        assert rotation.recording == recording
        assert rotation.x == input_data.rotation.x
        assert rotation.y == input_data.rotation.y
        assert rotation.z == input_data.rotation.z


def test_converts_all_resampled_joint_states(converter, input_data, recording):
    converter.resampler.resample.return_value = [
        Sample(data=input_data, timestamp=0.0),
        Sample(data=input_data, timestamp=1.0),
    ]

    models = converter.convert_to_model(input_data, 1.13, recording)

    assert len(models.joint_states) == 2
    assert models.joint_states[0].stamp == 0.0
    assert models.joint_states[1].stamp == 1.0

    for joint_state in models.joint_states:
        assert joint_state.recording == recording
        assert joint_state.r_shoulder_pitch == 0.0
        assert joint_state.l_shoulder_pitch == pytest.approx(0, abs=1e-5)
        assert joint_state.r_hip_yaw == np.pi
        assert joint_state.l_hip_yaw == pytest.approx(np.pi, abs=1e-5)
        assert joint_state.head_tilt == 0.0


def test_converts_all_resampled_joint_commands(converter, input_data, recording):
    converter.resampler.resample.return_value = [
        Sample(data=input_data, timestamp=0.0),
        Sample(data=input_data, timestamp=1.0),
    ]

    models = converter.convert_to_model(input_data, 1.13, recording)

    assert len(models.joint_commands) == 2
    assert models.joint_commands[0].stamp == 0.0
    assert models.joint_commands[1].stamp == 1.0

    for joint_command in models.joint_commands:
        assert joint_command.recording == recording
        assert joint_command.r_shoulder_pitch == 0.0
        assert joint_command.l_shoulder_pitch == pytest.approx(0, abs=1e-5)
        assert joint_command.r_hip_yaw == np.pi
        assert joint_command.l_hip_yaw == pytest.approx(np.pi, abs=1e-5)
        assert joint_command.head_tilt == 0.0


@pytest.fixture
def input_data(imu_msg, joint_position_msg, joint_command_msg):
    input = InputData()
    input.rotation = imu_msg
    input.joint_state = joint_position_msg
    input.joint_command = joint_command_msg

    return input


@pytest.fixture
def converter():
    resampler = Mock(PreviousInterpolationResampler)
    resampler.resample.return_value = []

    return SyncedDataConverter(resampler)


@pytest.fixture
def recording():
    return Recording(
        allow_public=True,
        team_name="Bit-Bots",
        robot_type="Wolfgang-OP",
        location="RoboCup2024",
        simulated=False,
    )
