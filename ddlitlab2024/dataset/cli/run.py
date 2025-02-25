#!/usr/bin/env python3

import typing

if typing.TYPE_CHECKING:
    from argparse import Namespace

import os
import sys
from pathlib import Path

from rich.console import Console

from ddlitlab2024 import DEFAULT_RESAMPLE_RATE_HZ, IMAGE_MAX_RESAMPLE_RATE_HZ, __version__
from ddlitlab2024.dataset import logger
from ddlitlab2024.dataset.cli.args import CLIArgs, CLICommand, DBCommand, ImportType
from ddlitlab2024.dataset.converters.game_state_converter import GameStateConverter
from ddlitlab2024.dataset.converters.image_converter import BHumanImageConverter, BitbotsImageConverter, ImageConverter
from ddlitlab2024.dataset.converters.synced_data_converter import SyncedDataConverter
from ddlitlab2024.dataset.db import Database
from ddlitlab2024.dataset.imports.model_importer import ImportStrategy
from ddlitlab2024.dataset.resampling.max_rate_resampler import MaxRateResampler
from ddlitlab2024.dataset.resampling.original_rate_resampler import OriginalRateResampler
from ddlitlab2024.dataset.resampling.previous_interpolation_resampler import PreviousInterpolationResampler

err_console = Console(stderr=True)


def main():
    debug_mode = os.getenv("LOGLEVEL") == "DEBUG"

    try:
        logger.debug("Parsing CLI args...")
        args: Namespace = CLIArgs().parse_args()
        logger.debug(f"Parsed CLI args: {args}")

        if args.version:
            logger.info(f"running ddlitlab2024 CLI v{__version__}")
            sys.exit(0)

        should_create_schema = args.command == CLICommand.DB and args.db_command == DBCommand.CREATE_SCHEMA
        db = Database(args.db_path).create_session(create_schema=should_create_schema)

        match args.command:
            case CLICommand.DB:
                match args.db_command:
                    case DBCommand.RECORDING2MCAP:
                        from ddlitlab2024.dataset.recording2mcap import recording2mcap

                        recording2mcap(db.session, args.recording, args.output_dir)

                    case DBCommand.DUMMY_DATA:
                        from ddlitlab2024.dataset.dummy_data import insert_dummy_data

                        insert_dummy_data(db.session, args.num_recordings, args.num_samples_per_rec, args.image_step)

            case CLICommand.IMPORT:
                from ddlitlab2024.dataset.imports.model_importer import ImportMetadata, ModelImporter

                import_strategy: ImportStrategy
                import_path: Path = Path(args.file)
                upper_image_converter: ImageConverter
                location: str = args.location

                match args.type:
                    case ImportType.BIT_BOTS:
                        from ddlitlab2024.dataset.imports.strategies.bit_bots import BitBotsImportStrategy

                        metadata = ImportMetadata(
                            allow_public=True,
                            team_name="Bit-Bots",
                            robot_type="Wolfgang-OP",
                            location=location,
                            simulated=False,
                        )
                        upper_image_converter = BitbotsImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        game_state_converter = GameStateConverter(OriginalRateResampler())
                        synced_data_converter = SyncedDataConverter(
                            PreviousInterpolationResampler(DEFAULT_RESAMPLE_RATE_HZ)
                        )
                        import_strategy = BitBotsImportStrategy(
                            metadata, upper_image_converter, game_state_converter, synced_data_converter
                        )

                    case ImportType.B_HUMAN:
                        from ddlitlab2024.dataset.imports.strategies.b_human import BHumanImportStrategy

                        metadata = ImportMetadata(
                            allow_public=False,
                            team_name="B-Human",
                            robot_type="NAO6",
                            location=location,
                            simulated=False,
                        )
                        upper_image_converter = BHumanImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        lower_image_converter = BHumanImageConverter(MaxRateResampler(IMAGE_MAX_RESAMPLE_RATE_HZ))
                        game_state_converter = GameStateConverter(OriginalRateResampler())
                        synced_data_converter = SyncedDataConverter(
                            PreviousInterpolationResampler(DEFAULT_RESAMPLE_RATE_HZ)
                        )

                        import_strategy = BHumanImportStrategy(
                            metadata,
                            upper_image_converter,
                            lower_image_converter,
                            game_state_converter,
                            synced_data_converter,
                            args.caching,
                            args.video,
                        )

                    case _:
                        raise ValueError(f"Unknown import type: {args.type}")

                logger.info(f"Importing file '{import_path}' to database...")
                importer = ModelImporter(db, import_strategy)
                importer.import_to_db(import_path)

        sys.exit(0)
    except Exception as e:
        logger.error(e)
        err_console.print_exception(show_locals=debug_mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
