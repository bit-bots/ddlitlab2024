import sys
from argparse import ArgumentParser, Namespace
from enum import Enum
from pathlib import Path

from ddlitlab2024 import DB_PATH
from ddlitlab2024.dataset.errors import CLIArgumentError


class ImportType(str, Enum):
    BIT_BOTS = "bit-bots"
    B_HUMAN = "b-human"


class CLICommand(str, Enum):
    DB = "db"
    IMPORT = "import"


class DBCommand(str, Enum):
    CREATE_SCHEMA = "create-schema"
    DUMMY_DATA = "dummy-data"
    RECORDING2MCAP = "recording2mcap"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class CLIArgs:
    def __init__(self):
        self.parser = ArgumentParser(description="ddlitlab dataset CLI")
        self.set_global_args(self.parser)

        subparsers = self.parser.add_subparsers(dest="command", help="Command to run")
        self.add_import_command_parser(subparsers)
        self.add_db_command_parser(subparsers)

    def set_global_args(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Dry run")
        parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to the sqlite database file")
        parser.add_argument("--version", action="store_true", help="Print version and exit")

    def add_db_command_parser(self, subparsers):
        self.db_parser = subparsers.add_parser(CLICommand.DB.value, help="Database management commands")
        db_subcommand_parser = self.db_parser.add_subparsers(dest="db_command", help="Database command")

        db_subcommand_parser.add_parser(
            DBCommand.CREATE_SCHEMA.value, help="Create the base database schema, if it doesn't exist"
        )

        # db dummy-data subcommand
        dummy_data_subparser = db_subcommand_parser.add_parser(
            DBCommand.DUMMY_DATA.value, help="Insert dummy data into the database"
        )
        dummy_data_subparser.add_argument(
            "-n", "--num_recordings", type=int, default=10, help="Number of recordings to insert"
        )
        dummy_data_subparser.add_argument(
            "-s", "--num_samples_per_rec", type=int, default=72000, help="Number of samples per recording"
        )
        dummy_data_subparser.add_argument("-i", "--image_step", type=int, default=10, help="Step size for images")

        # db recording2mcap subcommand
        recording2mcap_subparser = db_subcommand_parser.add_parser(
            DBCommand.RECORDING2MCAP.value, help="Convert a recording to an mcap file"
        )
        recording2mcap_subparser.add_argument("recording", type=str, help="Recording to convert")
        recording2mcap_subparser.add_argument("output_dir", type=Path, help="Output directory to write to")

    def add_import_command_parser(self, subparsers):
        self.import_parser = subparsers.add_parser(CLICommand.IMPORT.value, help="Import data into the database")
        self.import_parser.add_argument("type", type=ImportType, help="Type of import to perform")
        self.import_parser.add_argument("file", type=Path, help="File to import")
        self.import_parser.add_argument("location", type=str, help="Location of the data")
        self.import_parser.add_argument("--caching", action="store_true", help="Enable file caching")
        self.import_parser.add_argument("--video", action="store_true", help="Show video while importing")

    def parse_args(self) -> Namespace:
        return self.validate_args(self.parser.parse_args())

    def validate_args(self, args: Namespace) -> Namespace:
        if args.command == CLICommand.IMPORT.value:
            self.import_validation(args)
        elif args.command == CLICommand.DB.value:
            self.db_validation(args)

        return args

    def import_validation(self, args):
        if not args.file.exists():
            raise CLIArgumentError(f"File does not exist: {args.file}")

        if args.type == ImportType.BIT_BOTS and not args.file.suffix == ".mcap":
            raise CLIArgumentError(f"Rosbag import file not '*.mcap': {args.file}")

    def db_validation(self, args):
        if args.db_command not in DBCommand.values():
            self.print_help_and_exit(self.db_parser, exit_code=1)

        if not args.db_path.exists() and not args.db_command == DBCommand.CREATE_SCHEMA.value:
            raise CLIArgumentError(
                f"Database file does not exist: {args.db_path}. Run 'db create-schema' to create the database."
            )

    def print_help_and_exit(self, parser, exit_code: int = 0):
        parser.print_help()
        sys.exit(exit_code)
