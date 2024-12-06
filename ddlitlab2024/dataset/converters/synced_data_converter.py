from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.models import JointCommands, JointStates, Recording, Rotation
from ddlitlab2024.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler
from ddlitlab2024.utils.utils import camelcase_to_snakecase, shift_radian_to_positive_range


def joints_dict_from_msg_data(joints_data: list[tuple[str, float]]) -> dict[str, float]:
    joints_dict = {}

    for name, position in joints_data:
        key = camelcase_to_snakecase(name)
        value = shift_radian_to_positive_range(position)
        joints_dict[key] = value

    return joints_dict


class SyncedDataConverter(Converter):
    def __init__(self, resampler: PreviousInterpolationResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data: InputData, recording: Recording):
        pass

    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        assert data.joint_state is not None, "joint_states are required in synced resampling data"
        assert data.joint_command is not None, "joint_commands are required in synced resampling data"
        assert data.rotation is not None, "IMU rotation is required in synced resampling data"

        models = ModelData()

        for sample in self.resampler.resample(data, relative_timestamp):
            models.rotations.append(self._create_rotation(sample.data.rotation, sample.timestamp, recording))
            models.joint_states.append(self._create_joint_states(sample.data.joint_state, sample.timestamp, recording))
            models.joint_commands.append(
                self._create_joint_commands(sample.data.joint_command, sample.timestamp, recording)
            )

        return models

    def _create_rotation(self, msg, sampling_timestamp: float, recording: Recording) -> Rotation:
        return Rotation(
            stamp=sampling_timestamp,
            recording=recording,
            x=msg.x,
            y=msg.y,
            z=msg.z,
            w=msg.w,
        )

    def _create_joint_states(self, msg, sampling_timestamp: float, recording: Recording) -> JointStates:
        if msg is None:
            return JointStates(stamp=sampling_timestamp, recording=recording)
        else:
            joint_states_data = list(zip(msg.name, msg.position))

            return JointStates(
                stamp=sampling_timestamp, recording=recording, **joints_dict_from_msg_data(joint_states_data)
            )

    def _create_joint_commands(self, msg, sampling_timestamp: float, recording: Recording) -> JointCommands:
        if msg is None:
            return JointCommands(stamp=sampling_timestamp, recording=recording)
        else:
            joint_commands_data = list(zip(msg.joint_names, msg.positions))

            return JointCommands(
                stamp=sampling_timestamp, recording=recording, **joints_dict_from_msg_data(joint_commands_data)
            )
