import pytest

from ddlitlab2024.dataset.imports.data import InputData
from ddlitlab2024.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler

SAMPLE_RATE_HZ = 50


def test_resampling_initial_data(resampler, input_data):
    samples = resampler.resample(input_data, 0.0)

    assert len(samples) == 1
    assert samples[0].data == input_data
    assert samples[0].timestamp == 0.0


def test_resampling_before_next_sampling_step(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.01)

    assert samples == []


def test_resampling_at_next_sampling_step(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.02)

    assert len(samples) == 1
    assert samples[0].data == later_input_data
    assert samples[0].timestamp == 0.02


def test_resampling_after_next_sampling_step(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.03)

    assert len(samples) == 1
    assert samples[0].data == input_data
    assert samples[0].timestamp == 0.02


def test_resampling_multiple_steps(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.04)

    assert len(samples) == 2

    assert samples[0].data == input_data
    assert samples[0].timestamp == 0.02

    assert samples[1].data == later_input_data
    assert samples[1].timestamp == 0.04


def test_resampling_multiple_steps_after_next_sampling_step(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    resampler.resample(later_input_data, 0.01)
    samples = resampler.resample(input_data, 0.05)

    assert len(samples) == 2

    assert samples[0].data == later_input_data
    assert samples[0].timestamp == 0.02

    assert samples[1].data == later_input_data
    assert samples[1].timestamp == 0.04


@pytest.fixture
def resampler() -> PreviousInterpolationResampler:
    return PreviousInterpolationResampler(SAMPLE_RATE_HZ)


@pytest.fixture
def input_data() -> InputData:
    return InputData(joint_state="joint_state", rotation="rotation")


@pytest.fixture
def later_input_data() -> InputData:
    return InputData(image="image", game_state="game_state")
