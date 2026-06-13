import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_MASTER = DATA_DIR / "master"

# Create directories if they don't exist
for path in [DATA_RAW, DATA_PROCESSED, DATA_MASTER]:
    path.mkdir(parents=True, exist_ok=True)


# Global random seed for reproducibility
RANDOM_SEED = 42

# Contest Scoring Logic
# Dict defines points for: Exact Score, Correct Result, Incorrect
STAGE_SCORING = {
    "Group Stage": {"exact": 100, "result": 50, "incorrect": 0},
    "Round of 32": {"exact": 100, "result": 50, "incorrect": 0},
    "Round of 16": {"exact": 150, "result": 75, "incorrect": 0},
    "Quarter-final": {"exact": 200, "result": 100, "incorrect": 0},
    "Semi-final": {"exact": 300, "result": 150, "incorrect": 0},
    "Final / 3rd Place": {"exact": 400, "result": 200, "incorrect": 0},
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT/logs / "oracle.log"),
        logging.StreamHandler()
    ]
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
