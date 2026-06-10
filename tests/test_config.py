import pytest
from pathlib import Path
from src.config import PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, DATA_MASTER, RANDOM_SEED, STAGE_SCORING

def test_paths_exist():
    """Verify that all core directories were created successfully."""
    assert PROJECT_ROOT.exists()
    assert DATA_RAW.exists()
    assert DATA_PROCESSED.exists()
    assert DATA_MASTER.exists()

def test_random_seed():
    """Ensure global random seed is set to 42 for reproducibility."""
    assert RANDOM_SEED == 42

def test_scoring_logic():
    """Verify the scoring rules defined in RULE-AND-SYSTEM.md."""
    assert "Group Stage" in STAGE_SCORING
    assert STAGE_SCORING["Group Stage"]["exact"] == 100
    assert STAGE_SCORING["Final / 3rd Place"]["exact"] == 400
