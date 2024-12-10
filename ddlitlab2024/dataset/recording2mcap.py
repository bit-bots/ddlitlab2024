import json
import shutil
import sys
from pathlib import Path

from sqlalchemy.orm import Session
from transforms3d.euler import quat2euler

from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.models import Recording, stamp_to_nanoseconds, stamp_to_seconds_nanoseconds

try:
    import rosbag2_py
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Quaternion, Vector3
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Header, String
except ImportError:
    logger.error(
        "Failed to import ROS 2 packages. These are necessary to convert recordings to .mcap files. "
        "Make sure ROS 2 installed and sourced."
    )
    sys.exit(1)


def get_recording(db_session: Session, recording_id_or_filename: str | int) -> Recording:
    """Get the recording from the input string or integer

    param db: The database
    param recording_id_or_filename: The recording ID or original filename
    raises ValueError: If the recording does not exist
    return: The recording
    """
    if isinstance(recording_id_or_filename, int) or recording_id_or_filename.isdigit():
        # Verify that the recording exists
        recording_id = int(recording_id_or_filename)
        recording = db_session.query(Recording).get(recording_id)
        if recording is None:
            raise ValueError(f"Recording '{recording_id}' not found")
        return recording
    elif isinstance(recording_id_or_filename, str):
        recording = db_session.query(Recording).filter(Recording.original_file == recording_id_or_filename).first()
        if recording is None:
            raise ValueError(f"Recording with original filename '{recording_id_or_filename}' not found")
        return recording
    else:
        raise TypeError("Recording ID must be an integer or string")


def get_writer(output_dir: Path) -> rosbag2_py.SequentialWriter:
    """Get the mcap writer.

    param output_dir: The output directory
    return: The mcap writer
    """
    if output_dir.exists():
        # Ask the user if they want to overwrite the existing file
        if not input(f"Output directory '{output_dir}' already exists. Overwrite? (y/n): ").lower().startswith("y"):
            logger.info("Exiting")
            sys.exit(0)
        # Remove the existing directory
        shutil.rmtree(output_dir)

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_dir), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    return writer


def write_recording_info(recording: Recording, writer: rosbag2_py.SequentialWriter) -> None:
    """Write the recording info as a JSON encoded String message to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/recording", type="std_msgs/msg/String", serialization_format="cdr")
    )

    # Write recording info
    logger.info("Writing recording info")
    recording_info_msg = String(
        data=json.dumps(
            {
                "id": recording._id,
                "allow_public": recording.allow_public,
                "original_file": recording.original_file,
                "team_name": recording.team_name,
                "team_color": recording.team_color,
                "robot_type": recording.robot_type,
                "start_time": str(recording.start_time),
                "location": recording.location,
                "simulated": recording.simulated,
                "img_width": recording.img_width,
                "img_height": recording.img_height,
                "img_width_scaling": recording.img_width_scaling,
                "img_height_scaling": recording.img_height_scaling,
                "num_images": len(recording.images),
                "num_rotations": len(recording.rotations),
                "num_joint_states": len(recording.joint_states),
                "num_joint_commands": len(recording.joint_commands),
                "num_game_states": len(recording.game_states),
            }
        )
    )
    writer.write("/recording", serialize_message(recording_info_msg), 0)


def write_images(recording: Recording, writer: rosbag2_py.SequentialWriter) -> None:
    """Write the images to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/image", type="sensor_msgs/msg/Image", serialization_format="cdr")
    )

    # Write images
    logger.info("Writing images")
    for image in recording.images:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(image.stamp)
        image_msg = Image(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="camera_optical"),
            height=recording.img_height,
            width=recording.img_width,
            encoding="rgb8",
            is_bigendian=0,
            step=recording.img_width * 3,
            data=image.data,
        )
        writer.write("/image", serialize_message(image_msg), stamp_to_nanoseconds(image.stamp))


def write_rotations(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the rotations to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topics
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/rotation", type="geometry_msgs/msg/Quaternion", serialization_format="cdr")
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/rotation/euler", type="geometry_msgs/msg/Vector3", serialization_format="cdr")
    )

    # Write rotations
    logger.info("Writing rotations")
    for rotation in recording.rotations:
        rotation_msg = Quaternion(
            x=rotation.x,
            y=rotation.y,
            z=rotation.z,
            w=rotation.w,
        )
        writer.write("/rotation", serialize_message(rotation_msg), stamp_to_nanoseconds(rotation.stamp))

        # Convert quaternion to euler angles
        ax, ay, az = quat2euler([rotation.w, rotation.x, rotation.y, rotation.z], axes="sxyz")
        euler = Vector3(
            x=ax,
            y=ay,
            z=az,
        )
        writer.write("/rotation/euler", serialize_message(euler), stamp_to_nanoseconds(rotation.stamp))


def write_joint_states(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the joint states to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/joint_states", type="sensor_msgs/msg/JointState", serialization_format="cdr")
    )

    # Write joint states
    logger.info("Writing joint states")
    for joint_state in recording.joint_states:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(joint_state.stamp)
        joints: list[tuple[str, float]] = [
            ("r_shoulder_pitch", joint_state.r_shoulder_pitch),
            ("l_shoulder_pitch", joint_state.l_shoulder_pitch),
            ("r_shoulder_roll", joint_state.r_shoulder_roll),
            ("l_shoulder_roll", joint_state.l_shoulder_roll),
            ("r_elbow", joint_state.r_elbow),
            ("l_elbow", joint_state.l_elbow),
            ("r_hip_yaw", joint_state.r_hip_yaw),
            ("l_hip_yaw", joint_state.l_hip_yaw),
            ("r_hip_roll", joint_state.r_hip_roll),
            ("l_hip_roll", joint_state.l_hip_roll),
            ("r_hip_pitch", joint_state.r_hip_pitch),
            ("l_hip_pitch", joint_state.l_hip_pitch),
            ("r_knee", joint_state.r_knee),
            ("l_knee", joint_state.l_knee),
            ("r_ankle_pitch", joint_state.r_ankle_pitch),
            ("l_ankle_pitch", joint_state.l_ankle_pitch),
            ("r_ankle_roll", joint_state.r_ankle_roll),
            ("l_ankle_roll", joint_state.l_ankle_roll),
            ("head_pan", joint_state.head_pan),
            ("head_tilt", joint_state.head_tilt),
        ]
        joint_state_msg = JointState(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="base_link"),
            name=[name for name, _ in joints],
            position=[position for _, position in joints],
            velocity=[0.0] * len(joints),
            effort=[0.0] * len(joints),
        )
        writer.write("/joint_states", serialize_message(joint_state_msg), stamp_to_nanoseconds(joint_state.stamp))


def write_joint_commands(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the joint commands to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/joint_commands", type="sensor_msgs/msg/JointState", serialization_format="cdr")
    )

    # Write joint commands
    logger.info("Writing joint commands")
    for joint_command in recording.joint_commands:
        seconds, nanoseconds = stamp_to_seconds_nanoseconds(joint_command.stamp)
        joints: list[tuple[str, float | None]] = [
            ("r_shoulder_pitch", joint_command.r_shoulder_pitch),
            ("l_shoulder_pitch", joint_command.l_shoulder_pitch),
            ("r_shoulder_roll", joint_command.r_shoulder_roll),
            ("l_shoulder_roll", joint_command.l_shoulder_roll),
            ("r_elbow", joint_command.r_elbow),
            ("l_elbow", joint_command.l_elbow),
            ("r_hip_yaw", joint_command.r_hip_yaw),
            ("l_hip_yaw", joint_command.l_hip_yaw),
            ("r_hip_roll", joint_command.r_hip_roll),
            ("l_hip_roll", joint_command.l_hip_roll),
            ("r_hip_pitch", joint_command.r_hip_pitch),
            ("l_hip_pitch", joint_command.l_hip_pitch),
            ("r_knee", joint_command.r_knee),
            ("l_knee", joint_command.l_knee),
            ("r_ankle_pitch", joint_command.r_ankle_pitch),
            ("l_ankle_pitch", joint_command.l_ankle_pitch),
            ("r_ankle_roll", joint_command.r_ankle_roll),
            ("l_ankle_roll", joint_command.l_ankle_roll),
            ("head_pan", joint_command.head_pan),
            ("head_tilt", joint_command.head_tilt),
        ]
        joint_command_msg = JointState(
            header=Header(stamp=Time(sec=seconds, nanosec=nanoseconds), frame_id="base_link"),
            name=[name for name, _ in joints],
            position=[position for _, position in joints],
            velocity=[0.0] * len(joints),
            effort=[0.0] * len(joints),
        )
        writer.write("/joint_commands", serialize_message(joint_command_msg), stamp_to_nanoseconds(joint_command.stamp))


def write_game_states(
    recording: Recording,
    writer: rosbag2_py.SequentialWriter,
) -> None:
    """Write the game states to the mcap file

    param recording: The recording
    param writer: The mcap writer
    """
    # Create topic
    writer.create_topic(
        rosbag2_py.TopicMetadata(name="/game_state", type="std_msgs/msg/String", serialization_format="cdr")
    )

    # Write game states
    logger.info("Writing game states")
    for game_state in recording.game_states:
        game_state_msg = String(data=game_state.state)
        writer.write("/game_state", serialize_message(game_state_msg), stamp_to_nanoseconds(game_state.stamp))


def recording2mcap(db_session: Session, recording_id_or_filename: str | int, output: Path) -> None:
    """Convert a recording to an mcap file

    param db: The database
    param recording_id_or_filename: The recording ID or original filename
    param output: The output mcap file
    """
    recording = get_recording(db_session, recording_id_or_filename)
    logger.info(f"Converting recording '{recording._id}' to mcap file '{output}'")

    writer = get_writer(output)
    write_recording_info(recording, writer)
    write_images(recording, writer)
    write_rotations(recording, writer)
    write_joint_states(recording, writer)
    write_joint_commands(recording, writer)
    write_game_states(recording, writer)

    logger.info(f"Recording '{recording._id}' converted to mcap file '{output}'")
