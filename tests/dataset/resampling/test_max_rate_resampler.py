import pytest

from ddlitlab2024.dataset.imports.data import InputData
from ddlitlab2024.dataset.resampling.max_rate_resampler import MaxRateResampler

MAX_SAMPLE_RATE_HZ = 50


def test_resampling_initial_data(resampler, input_data):
    samples = resampler.resample(input_data, 0.0)

    assert len(samples) == 1
    assert samples[0].data == input_data
    assert samples[0].timestamp == 0.0


def test_resampling_next_data(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.02)

    assert len(samples) == 1
    assert samples[0].data == later_input_data
    assert samples[0].timestamp == 0.02


def test_resampling_higher_than_max_rate(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 0.01)

    assert samples == []


def test_resampling_low_sampling_rate(resampler, input_data, later_input_data):
    resampler.resample(input_data, 0.0)
    samples = resampler.resample(later_input_data, 1.0)

    assert len(samples) == 1
    assert samples[0].data == later_input_data
    assert samples[0].timestamp == 1.0


@pytest.fixture
def resampler() -> MaxRateResampler:
    return MaxRateResampler(max_sample_rate_hz=MAX_SAMPLE_RATE_HZ)


@pytest.fixture
def input_data() -> InputData:
    return InputData(joint_state="joint_state", rotation="rotation")


@pytest.fixture
def later_input_data() -> InputData:
    return InputData(image="image", game_state="game_state")
