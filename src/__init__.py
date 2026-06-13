import os
import logging
import pathlib

logger = logging.getLogger(__name__)


def _load_dotenv_file(path: str) -> None:
    try:
        with open(path, 'r') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Respect existing environment variables
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as exc:
        logger.debug(f"Failed to read .env at {path}: {exc}")


# 1) Try python-dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env using python-dotenv")
except Exception:
    # 2) Fallback: look for .env at repo root and parse it
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    dotenv_path = repo_root / '.env'
    if dotenv_path.exists():
        _load_dotenv_file(str(dotenv_path))
        logger.info(f"Loaded .env from {dotenv_path}")
    else:
        logger.debug("python-dotenv not installed and .env file not found; skipping .env load")
