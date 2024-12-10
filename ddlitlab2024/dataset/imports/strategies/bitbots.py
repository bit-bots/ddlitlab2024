from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import transforms3d as t3d
from mcap.reader import make_reader
from mcap.summary import Summary
from mcap_ros2.decoder import DecoderFactory

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.converters.converter import Converter
from ddlitlab2024.dataset.converters.game_state_converter import GameStateConverter
from ddlitlab2024.dataset.converters.image_converter import ImageConverter
from ddlitlab2024.dataset.converters.synced_data_converter import SyncedDataConverter
from ddlitlab2024.dataset.imports.data import InputData, ModelData
from ddlitlab2024.dataset.imports.model_importer import ImportMetadata, ImportStrategy
from ddlitlab2024.dataset.models import DEFAULT_IMG_SIZE, Recording, Rotation

USED_TOPICS = [
    "/DynamixelController/command",
    "/camera/image_proc",
    "/camera/image_to_record",
    "/gamestate",
    "/imu/data",
    "/joint_states",
    "/tf",
]


class BitBotsImportStrategy(ImportStrategy):
    def __init__(
        self,
        metadata: ImportMetadata,
        image_converter: ImageConverter,
        game_state_converter: GameStateConverter,
        synced_data_converter: SyncedDataConverter,
    ):
        self.metadata = metadata

        self.image_converter = image_converter
        self.game_state_converter = game_state_converter
        self.synced_data_converter = synced_data_converter

        self.model_data = ModelData()

    def convert_to_model_data(self, file_path: Path) -> ModelData:
        with self._mcap_reader(file_path) as reader:
            summary: Summary | None = reader.get_summary()

            if summary is None:
                logger.error("No summary found in the MCAP file, skipping processing.")
                return self.model_data

            first_used_msg_time = None
            last_messages_by_topic = InputData()

            self.model_data.recording = self._create_recording(summary, file_path)

            self._log_debug_info(summary, self.model_data.recording)

            # Check if we got any imu messages
            has_imu_data = any(channel.topic == "/imu/data" for channel in summary.channels.values())

            for _, channel, message, ros_msg in reader.iter_decoded_messages(topics=USED_TOPICS):
                converter: Converter | None = None

                match channel.topic:
                    case "/gamestate":
                        last_messages_by_topic.game_state = ros_msg
                        converter = self.game_state_converter
                    case "/camera/image_proc" | "/camera/image_to_record":
                        last_messages_by_topic.image = ros_msg
                        converter = self.image_converter
                    case "/joint_states":
                        last_messages_by_topic.joint_state = ros_msg
                        converter = self.synced_data_converter
                    case "/DynamixelController/command":
                        last_messages_by_topic.joint_command = ros_msg
                        converter = self.synced_data_converter
                    case "/imu/data":
                        assert has_imu_data, "IMU data is not expected in this MCAP file"
                        last_messages_by_topic.rotation = ros_msg.orientation
                        converter = self.synced_data_converter
                    case "/tf":
                        if not has_imu_data:
                            for tf_msg in ros_msg.transforms:
                                if tf_msg.child_frame_id == "base_footprint" and tf_msg.header.frame_id == "base_link":
                                    quat = tf_msg.transform.rotation
                                    # Invert the quaternion to get the rotation from base_footprint
                                    # to base_link instead of the other way around
                                    # This is necessary to get pitch and roll angles in the correct frame
                                    w, x, y, z = t3d.quaternions.qinverse([quat.w, quat.x, quat.y, quat.z])
                                    last_messages_by_topic.rotation = Rotation(x=x, y=y, z=z, w=w)
                                    converter = self.synced_data_converter
                    case _:
                        logger.warning(f"Unhandled topic: {channel.topic} without conversion. Skipping...")

                if self._is_all_synced_data_available(last_messages_by_topic):
                    if first_used_msg_time is None:
                        first_used_msg_time = message.publish_time
                        self._initial_conversion(last_messages_by_topic)
                    else:
                        relative_msg_timestamp = (message.publish_time - first_used_msg_time) / 1e9
                        if converter:
                            self._create_models(converter, last_messages_by_topic, relative_msg_timestamp)

            return self.model_data

    def _initial_conversion(self, data: InputData):
        assert self._is_all_synced_data_available(data), "All synced data must be available to create initial models"

        first_timestamp = 0.0

        if data.game_state:
            self._create_models(self.game_state_converter, data, first_timestamp)

        self._create_models(self.synced_data_converter, data, first_timestamp)

    def _create_models(self, converter: Converter, data: InputData, relative_timestamp: float) -> ModelData:
        assert self.model_data.recording is not None, "Recording must be defined to create child models"

        converter.populate_recording_metadata(data, self.model_data.recording)
        model_data = converter.convert_to_model(data, relative_timestamp, self.model_data.recording)
        self.model_data = self.model_data.merge(model_data)

        return self.model_data

    def _is_all_synced_data_available(self, data: InputData) -> bool:
        # @TODO: add check for IMU data, when tf conversion to rotation is implemented
        # return data.joint_command is not None and data.joint_state is not None and data.rotation is not None
        return data.joint_command is not None and data.joint_state is not None

    def _create_recording(self, summary: Summary, mcap_file_path: Path) -> Recording:
        start_timestamp, end_timestamp = self._extract_timeframe(summary)

        return Recording(
            allow_public=self.metadata.allow_public,
            original_file=mcap_file_path.name,
            team_name=self.metadata.team_name,
            robot_type=self.metadata.robot_type,
            start_time=datetime.fromtimestamp(start_timestamp / 1e9),
            end_time=datetime.fromtimestamp(end_timestamp / 1e9),
            location=self.metadata.location,
            simulated=self.metadata.simulated,
            img_width=DEFAULT_IMG_SIZE[0],
            img_height=DEFAULT_IMG_SIZE[1],
            # needs to be overwritten when processing images
            img_width_scaling=0.0,
            img_height_scaling=0.0,
        )

    def _extract_timeframe(self, summary: Summary) -> tuple[int, int]:
        first_msg_start_time = None
        last_msg_end_time = None

        for chunk_index in summary.chunk_indexes:
            if first_msg_start_time is None or chunk_index.message_start_time < first_msg_start_time:
                first_msg_start_time = chunk_index.message_start_time
            if last_msg_end_time is None or chunk_index.message_end_time > last_msg_end_time:
                last_msg_end_time = chunk_index.message_end_time

        assert first_msg_start_time is not None, "No start time found in the MCAP file"
        assert last_msg_end_time is not None, "No end time found in the MCAP file"

        return first_msg_start_time, last_msg_end_time

    @contextmanager
    def _mcap_reader(self, mcap_file_path: Path):
        with open(mcap_file_path, "rb") as f:
            yield make_reader(f, decoder_factories=[DecoderFactory()])

    def _log_debug_info(self, summary: Summary, recording: Recording):
        log_message = f"Processing rosbag: {recording.original_file} - {recording.team_name}"
        if recording.location:
            log_message += f" {recording.location}"
        if recording.start_time:
            log_message += f": {recording.start_time}"

        available_topics = [channel.topic for channel in summary.channels.values()]
        log_message += f"\nAvailable topics: {available_topics}"
        log_message += f"\nUsed topics: {USED_TOPICS}"

        logger.info(log_message)
