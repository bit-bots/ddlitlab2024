from enum import Enum

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.models import GameState, Recording, RobotState, TeamColor
from ddlitlab2024.dataset.resampling.original_rate_resampler import OriginalRateResampler


class GameStateMessage(int, Enum):
    INITIAL = 0
    READY = 1
    SET = 2
    PLAYING = 3
    FINISHED = 4


class GameStateConverter(Converter):
    def __init__(self, resampler: OriginalRateResampler) -> None:
        self.resampler = resampler

    def populate_recording_metadata(self, data, recording: Recording):
        team_color = TeamColor.BLUE if data.game_state.team_color == 0 else TeamColor.RED
        if recording.team_color is None:
            recording.team_color = team_color

        team_color_changed = recording.team_color != team_color

        if team_color_changed:
            logger.warning("The team color changed, during one recording! This will be ignored.")

    def convert_to_model(self, data: InputData, relative_timestamp: float, recording: Recording) -> ModelData:
        models = ModelData()

        for sample in self.resampler.resample(data, relative_timestamp):
            models.game_states.append(self._create_game_state(sample.data.game_state, sample.timestamp, recording))

        return models

    def _create_game_state(self, msg, sampling_timestamp: float, recording: Recording) -> GameState:
        return GameState(stamp=sampling_timestamp, recording=recording, state=self._robot_state_from_msg(msg))

    def _robot_state_from_msg(self, msg) -> RobotState:
        if msg.penalized:
            return RobotState.STOPPED

        match msg.game_state:
            case GameStateMessage.INITIAL:
                return RobotState.STOPPED
            case GameStateMessage.READY:
                return RobotState.POSITIONING
            case GameStateMessage.SET:
                return RobotState.STOPPED
            case GameStateMessage.PLAYING:
                return RobotState.PLAYING
            case GameStateMessage.FINISHED:
                return RobotState.STOPPED
            case _:
                return RobotState.UNKNOWN
