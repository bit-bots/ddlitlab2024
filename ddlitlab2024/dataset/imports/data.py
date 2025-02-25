from dataclasses import dataclass, field
from typing import Any

from ddlitlab2024.dataset.models import GameState, Image, JointCommands, JointStates, Recording, Rotation
from ddlitlab2024.utils.utils import camelcase_to_snakecase, shift_radian_to_positive_range


def joints_dict_from_msg_data(joints_data: list[tuple[str, float]]) -> dict[str, float]:
    joints_dict = {}

    for name, position in joints_data:
        key = camelcase_to_snakecase(name)
        value = shift_radian_to_positive_range(position)
        joints_dict[key] = value

    return joints_dict


@dataclass
class ImportMetadata:
    allow_public: bool
    team_name: str
    robot_type: str
    location: str
    simulated: bool


@dataclass
class InputData:
    image: Any = None
    lower_image: Any = None
    game_state: Any = None
    joint_state: Any = None
    rotation: Any = None

    # as we are not always sending joint commands for all joints at once
    # we need to separate them here, to enable resampling on a per joint basis
    r_shoulder_pitch_command: Any = None
    l_shoulder_pitch_command: Any = None
    r_shoulder_roll_command: Any = None
    l_shoulder_roll_command: Any = None
    r_elbow_command: Any = None
    l_elbow_command: Any = None
    r_hip_yaw_command: Any = None
    l_hip_yaw_command: Any = None
    r_hip_roll_command: Any = None
    l_hip_roll_command: Any = None
    r_hip_pitch_command: Any = None
    l_hip_pitch_command: Any = None
    r_knee_command: Any = None
    l_knee_command: Any = None
    r_ankle_pitch_command: Any = None
    l_ankle_pitch_command: Any = None
    r_ankle_roll_command: Any = None
    l_ankle_roll_command: Any = None
    head_pan_command: Any = None
    head_tilt_command: Any = None

    @property
    def joint_command(self):
        return {
            "r_shoulder_pitch": self.r_shoulder_pitch_command,
            "l_shoulder_pitch": self.l_shoulder_pitch_command,
            "r_shoulder_roll": self.r_shoulder_roll_command,
            "l_shoulder_roll": self.l_shoulder_roll_command,
            "r_elbow": self.r_elbow_command,
            "l_elbow": self.l_elbow_command,
            "r_hip_yaw": self.r_hip_yaw_command,
            "l_hip_yaw": self.l_hip_yaw_command,
            "r_hip_roll": self.r_hip_roll_command,
            "l_hip_roll": self.l_hip_roll_command,
            "r_hip_pitch": self.r_hip_pitch_command,
            "l_hip_pitch": self.l_hip_pitch_command,
            "r_knee": self.r_knee_command,
            "l_knee": self.l_knee_command,
            "r_ankle_pitch": self.r_ankle_pitch_command,
            "l_ankle_pitch": self.l_ankle_pitch_command,
            "r_ankle_roll": self.r_ankle_roll_command,
            "l_ankle_roll": self.l_ankle_roll_command,
            "head_pan": self.head_pan_command,
            "head_tilt": self.head_tilt_command,
        }

    @joint_command.setter
    def joint_command(self, msg):
        joint_commands_data = list(zip(msg.joint_names, msg.positions))

        for joint_name, command in joints_dict_from_msg_data(joint_commands_data).items():
            setattr(self, f"{joint_name}_command", command)


@dataclass
class ModelData:
    recording: Recording | None = None
    game_states: list[GameState] = field(default_factory=list)
    joint_states: list[JointStates] = field(default_factory=list)
    joint_commands: list[JointCommands] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)
    rotations: list[Rotation] = field(default_factory=list)

    def model_instances(self):
        return [self.recording] + self.game_states + self.joint_states + self.joint_commands + self.images

    def merge(self, other: "ModelData") -> "ModelData":
        self.game_states.extend(other.game_states)
        self.joint_states.extend(other.joint_states)
        self.joint_commands.extend(other.joint_commands)
        self.images.extend(other.images)
        return self
