"""Base dataset

Revision ID: 3f1574e89695
Revises:
Create Date: 2025-01-23 15:39:02.177099

"""

from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f1574e89695"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "Recording",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("allow_public", sa.Boolean(), nullable=False),
        sa.Column("original_file", sa.String(), nullable=False),
        sa.Column("team_name", sa.String(), nullable=False),
        sa.Column("team_color", sa.String(), nullable=True),
        sa.Column("robot_type", sa.String(), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        sa.Column("location", sa.String(), nullable=True),
        sa.Column("simulated", sa.Boolean(), nullable=False),
        sa.Column("img_width", sa.Integer(), nullable=False),
        sa.Column("img_height", sa.Integer(), nullable=False),
        sa.Column("img_width_scaling", sa.Float(), nullable=False),
        sa.Column("img_height_scaling", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "team_color IN ('BLUE', 'RED', 'YELLOW', 'BLACK', 'WHITE', 'GREEN', 'ORANGE', 'PURPLE', 'BROWN', 'GRAY')",
            name=op.f("ck_Recording_team_color_enum"),
        ),
        sa.CheckConstraint("end_time >= start_time", name=op.f("ck_Recording_end_time_ge_start_time")),
        sa.CheckConstraint("img_height > 0", name=op.f("ck_Recording_img_height_value")),
        sa.CheckConstraint("img_width > 0", name=op.f("ck_Recording_img_width_value")),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_Recording")),
    )

    op.create_table(
        "GameState",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stamp", sa.Float(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.CheckConstraint(
            "state IN ('PLAYING', 'POSITIONING', 'STOPPED', 'UNKNOWN')", name=op.f("ck_GameState_state_enum")
        ),
        sa.ForeignKeyConstraint(["recording_id"], ["Recording._id"], name=op.f("fk_GameState_recording_id_Recording")),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_GameState")),
        sa.Index(None, "recording_id", sa.asc("stamp")),
    )

    op.create_table(
        "Image",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stamp", sa.Float(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.CheckConstraint("stamp >= 0", name=op.f("ck_Image_stamp_value")),
        sa.ForeignKeyConstraint(["recording_id"], ["Recording._id"], name=op.f("fk_Image_recording_id_Recording")),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_Image")),
        sa.Index(None, "recording_id", sa.asc("stamp")),
    )

    op.create_table(
        "JointCommands",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stamp", sa.Float(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("RShoulderPitch", sa.Float(), nullable=False),
        sa.Column("LShoulderPitch", sa.Float(), nullable=False),
        sa.Column("RShoulderRoll", sa.Float(), nullable=False),
        sa.Column("LShoulderRoll", sa.Float(), nullable=False),
        sa.Column("RElbow", sa.Float(), nullable=False),
        sa.Column("LElbow", sa.Float(), nullable=False),
        sa.Column("RHipYaw", sa.Float(), nullable=False),
        sa.Column("LHipYaw", sa.Float(), nullable=False),
        sa.Column("RHipRoll", sa.Float(), nullable=False),
        sa.Column("LHipRoll", sa.Float(), nullable=False),
        sa.Column("RHipPitch", sa.Float(), nullable=False),
        sa.Column("LHipPitch", sa.Float(), nullable=False),
        sa.Column("RKnee", sa.Float(), nullable=False),
        sa.Column("LKnee", sa.Float(), nullable=False),
        sa.Column("RAnklePitch", sa.Float(), nullable=False),
        sa.Column("LAnklePitch", sa.Float(), nullable=False),
        sa.Column("RAnkleRoll", sa.Float(), nullable=False),
        sa.Column("LAnkleRoll", sa.Float(), nullable=False),
        sa.Column("HeadPan", sa.Float(), nullable=False),
        sa.Column("HeadTilt", sa.Float(), nullable=False),
        sa.CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()", name=op.f("ck_JointCommands_HeadPan_value")),
        sa.CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()", name=op.f("ck_JointCommands_HeadTilt_value")),
        sa.CheckConstraint(
            "LAnklePitch >= 0 AND LAnklePitch < 2 * pi()", name=op.f("ck_JointCommands_LAnklePitch_value")
        ),
        sa.CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()", name=op.f("ck_JointCommands_LAnkleRoll_value")),
        sa.CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()", name=op.f("ck_JointCommands_LElbow_value")),
        sa.CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()", name=op.f("ck_JointCommands_LHipPitch_value")),
        sa.CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()", name=op.f("ck_JointCommands_LHipRoll_value")),
        sa.CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()", name=op.f("ck_JointCommands_LHipYaw_value")),
        sa.CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()", name=op.f("ck_JointCommands_LKnee_value")),
        sa.CheckConstraint(
            "LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()", name=op.f("ck_JointCommands_LShoulderPitch_value")
        ),
        sa.CheckConstraint(
            "LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()", name=op.f("ck_JointCommands_LShoulderRoll_value")
        ),
        sa.CheckConstraint(
            "RAnklePitch >= 0 AND RAnklePitch < 2 * pi()", name=op.f("ck_JointCommands_RAnklePitch_value")
        ),
        sa.CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()", name=op.f("ck_JointCommands_RAnkleRoll_value")),
        sa.CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()", name=op.f("ck_JointCommands_RElbow_value")),
        sa.CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()", name=op.f("ck_JointCommands_RHipPitch_value")),
        sa.CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()", name=op.f("ck_JointCommands_RHipRoll_value")),
        sa.CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()", name=op.f("ck_JointCommands_RHipYaw_value")),
        sa.CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()", name=op.f("ck_JointCommands_RKnee_value")),
        sa.CheckConstraint(
            "RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()", name=op.f("ck_JointCommands_RShoulderPitch_value")
        ),
        sa.CheckConstraint(
            "RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()", name=op.f("ck_JointCommands_RShoulderRoll_value")
        ),
        sa.CheckConstraint("stamp >= 0", name=op.f("ck_JointCommands_stamp_value")),
        sa.ForeignKeyConstraint(
            ["recording_id"], ["Recording._id"], name=op.f("fk_JointCommands_recording_id_Recording")
        ),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_JointCommands")),
        sa.Index(None, "recording_id", sa.asc("stamp")),
    )

    op.create_table(
        "JointStates",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stamp", sa.Float(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("RShoulderPitch", sa.Float(), nullable=False),
        sa.Column("LShoulderPitch", sa.Float(), nullable=False),
        sa.Column("RShoulderRoll", sa.Float(), nullable=False),
        sa.Column("LShoulderRoll", sa.Float(), nullable=False),
        sa.Column("RElbow", sa.Float(), nullable=False),
        sa.Column("LElbow", sa.Float(), nullable=False),
        sa.Column("RHipYaw", sa.Float(), nullable=False),
        sa.Column("LHipYaw", sa.Float(), nullable=False),
        sa.Column("RHipRoll", sa.Float(), nullable=False),
        sa.Column("LHipRoll", sa.Float(), nullable=False),
        sa.Column("RHipPitch", sa.Float(), nullable=False),
        sa.Column("LHipPitch", sa.Float(), nullable=False),
        sa.Column("RKnee", sa.Float(), nullable=False),
        sa.Column("LKnee", sa.Float(), nullable=False),
        sa.Column("RAnklePitch", sa.Float(), nullable=False),
        sa.Column("LAnklePitch", sa.Float(), nullable=False),
        sa.Column("RAnkleRoll", sa.Float(), nullable=False),
        sa.Column("LAnkleRoll", sa.Float(), nullable=False),
        sa.Column("HeadPan", sa.Float(), nullable=False),
        sa.Column("HeadTilt", sa.Float(), nullable=False),
        sa.CheckConstraint("HeadPan >= 0 AND HeadPan < 2 * pi()", name=op.f("ck_JointStates_HeadPan_value")),
        sa.CheckConstraint("HeadTilt >= 0 AND HeadTilt < 2 * pi()", name=op.f("ck_JointStates_HeadTilt_value")),
        sa.CheckConstraint(
            "LAnklePitch >= 0 AND LAnklePitch < 2 * pi()", name=op.f("ck_JointStates_LAnklePitch_value")
        ),
        sa.CheckConstraint("LAnkleRoll >= 0 AND LAnkleRoll < 2 * pi()", name=op.f("ck_JointStates_LAnkleRoll_value")),
        sa.CheckConstraint("LElbow >= 0 AND LElbow < 2 * pi()", name=op.f("ck_JointStates_LElbow_value")),
        sa.CheckConstraint("LHipPitch >= 0 AND LHipPitch < 2 * pi()", name=op.f("ck_JointStates_LHipPitch_value")),
        sa.CheckConstraint("LHipRoll >= 0 AND LHipRoll < 2 * pi()", name=op.f("ck_JointStates_LHipRoll_value")),
        sa.CheckConstraint("LHipYaw >= 0 AND LHipYaw < 2 * pi()", name=op.f("ck_JointStates_LHipYaw_value")),
        sa.CheckConstraint("LKnee >= 0 AND LKnee < 2 * pi()", name=op.f("ck_JointStates_LKnee_value")),
        sa.CheckConstraint(
            "LShoulderPitch >= 0 AND LShoulderPitch < 2 * pi()", name=op.f("ck_JointStates_LShoulderPitch_value")
        ),
        sa.CheckConstraint(
            "LShoulderRoll >= 0 AND LShoulderRoll < 2 * pi()", name=op.f("ck_JointStates_LShoulderRoll_value")
        ),
        sa.CheckConstraint(
            "RAnklePitch >= 0 AND RAnklePitch < 2 * pi()", name=op.f("ck_JointStates_RAnklePitch_value")
        ),
        sa.CheckConstraint("RAnkleRoll >= 0 AND RAnkleRoll < 2 * pi()", name=op.f("ck_JointStates_RAnkleRoll_value")),
        sa.CheckConstraint("RElbow >= 0 AND RElbow < 2 * pi()", name=op.f("ck_JointStates_RElbow_value")),
        sa.CheckConstraint("RHipPitch >= 0 AND RHipPitch < 2 * pi()", name=op.f("ck_JointStates_RHipPitch_value")),
        sa.CheckConstraint("RHipRoll >= 0 AND RHipRoll < 2 * pi()", name=op.f("ck_JointStates_RHipRoll_value")),
        sa.CheckConstraint("RHipYaw >= 0 AND RHipYaw < 2 * pi()", name=op.f("ck_JointStates_RHipYaw_value")),
        sa.CheckConstraint("RKnee >= 0 AND RKnee < 2 * pi()", name=op.f("ck_JointStates_RKnee_value")),
        sa.CheckConstraint(
            "RShoulderPitch >= 0 AND RShoulderPitch < 2 * pi()", name=op.f("ck_JointStates_RShoulderPitch_value")
        ),
        sa.CheckConstraint(
            "RShoulderRoll >= 0 AND RShoulderRoll < 2 * pi()", name=op.f("ck_JointStates_RShoulderRoll_value")
        ),
        sa.CheckConstraint("stamp >= 0", name=op.f("ck_JointStates_stamp_value")),
        sa.ForeignKeyConstraint(
            ["recording_id"], ["Recording._id"], name=op.f("fk_JointStates_recording_id_Recording")
        ),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_JointStates")),
        sa.Index(None, "recording_id", sa.asc("stamp")),
    )

    op.create_table(
        "Rotation",
        sa.Column("_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("stamp", sa.Float(), nullable=False),
        sa.Column("recording_id", sa.Integer(), nullable=False),
        sa.Column("x", sa.Float(), nullable=False),
        sa.Column("y", sa.Float(), nullable=False),
        sa.Column("z", sa.Float(), nullable=False),
        sa.Column("w", sa.Float(), nullable=False),
        sa.CheckConstraint("stamp >= 0", name=op.f("ck_Rotation_stamp_value")),
        sa.CheckConstraint("w >= -1 AND w <= 1", name=op.f("ck_Rotation_w_value")),
        sa.CheckConstraint("x >= -1 AND x <= 1", name=op.f("ck_Rotation_x_value")),
        sa.CheckConstraint("y >= -1 AND y <= 1", name=op.f("ck_Rotation_y_value")),
        sa.CheckConstraint("z >= -1 AND z <= 1", name=op.f("ck_Rotation_z_value")),
        sa.ForeignKeyConstraint(["recording_id"], ["Recording._id"], name=op.f("fk_Rotation_recording_id_Recording")),
        sa.PrimaryKeyConstraint("_id", name=op.f("pk_Rotation")),
        sa.Index(None, "recording_id", sa.asc("stamp")),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("Rotation")
    op.drop_table("JointStates")
    op.drop_table("JointCommands")
    op.drop_table("Image")
    op.drop_table("GameState")
    op.drop_table("Recording")
    # ### end Alembic commands ###
