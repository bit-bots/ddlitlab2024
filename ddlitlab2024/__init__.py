import importlib.metadata
import os
import sys
from pathlib import Path
from uuid import UUID, uuid4

_project_name: str = "ddlitlab2024"
__version__: str = importlib.metadata.version(_project_name)

# Craft LOGGING PATH
# Get the log directory from the environment variable or use the default
_logging_dir: str = os.path.abspath(
    os.environ.get("DDLITLAB_LOG_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
)

# Verify that the log directory exists and create it if it doesn't
if not os.path.exists(_logging_dir):
    try:
        os.makedirs(_logging_dir)
    except OSError:
        print(f"ERROR: Failed to create log directory '{_logging_dir}'. Exiting.")
        sys.exit(1)

_logging_path: str = os.path.join(_logging_dir, f"{_project_name}.log")

# Create log file if it doesn't exist or verify that it is writable
if os.path.exists(_logging_path):
    if not os.access(_logging_path, os.W_OK):
        print(f"ERROR: Log file '{_logging_path}' is not writable. Exiting.")
        sys.exit(1)
else:
    try:
        open(_logging_path, "w").close()
        print(f"INFO: Created log file '{_logging_path}'.")
    except OSError:
        print(f"ERROR: Failed to open log file '{_logging_path}'. Exiting.")
        sys.exit(1)

LOGGING_PATH: str = _logging_path

SESSION_ID: UUID = uuid4()

DB_PATH: Path = Path(os.environ.get("DDLITLAB_DB_PATH", Path.joinpath(Path(__file__).parent, "dataset", "db.sqlite3")))

DEFAULT_RESAMPLE_RATE_HZ = 50
IMAGE_MAX_RESAMPLE_RATE_HZ = 10
