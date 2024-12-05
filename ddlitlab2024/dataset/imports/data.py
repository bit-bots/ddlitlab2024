from dataclasses import dataclass, field
from typing import Any

from ddlitlab2024.dataset.models import GameState, Image, JointCommands, JointStates, Recording, Rotation


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
    game_state: Any = None
    joint_state: Any = None
    joint_command: Any = None
    rotation: Any = None


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
