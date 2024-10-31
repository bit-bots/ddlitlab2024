import argparse
from enum import Enum
from pathlib import Path

from ddlitlab2024 import DB_PATH


class ImportType(str, Enum):
    ROS_BAG = "rosbag"


class CLICommand(str, Enum):
    DB = "db"
    IMPORT = "import"


class CLIArgs:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="ddlitlab dataset CLI")

        self.parser.add_argument("--dry-run", action="store_true", help="Dry run")
        self.parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to the sqlite database file")
        self.parser.add_argument("--version", action="store_true", help="Print version and exit")

        subparsers = self.parser.add_subparsers(dest="command", help="Command to run")
        # import_parser = subparsers.add_parser(CLICommand.IMPORT.value, help="Import data into the database")

        db_parser = subparsers.add_parser(CLICommand.DB.value, help="Database management commands")
        db_subcommand_parser = db_parser.add_subparsers(dest="db_command", help="Database command")

        db_subcommand_parser.add_parser("create-schema", help="Create the base database schema, if it doesn't exist.")
        recording2mcap_subparser = db_subcommand_parser.add_parser(
            "recording2mcap", help="Convert a recording to an mcap file"
        )
        recording2mcap_subparser.add_argument("recording", type=str, help="Recording to convert")
        recording2mcap_subparser.add_argument("output", type=Path, help="MCAP output file to write to")

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()
