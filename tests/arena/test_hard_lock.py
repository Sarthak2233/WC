import pytest
import os
import csv
from datetime import datetime, timedelta, timezone
from src.arena.hard_lock_engine import HardLockEngine, LockViolationError, MatchNotLockedError, ResultAlreadyExistsError

# Using a temp file for tests to avoid polluting actual ledger
LOCK_PATH = "tests/arena/test_lock.csv"
RESULTS_PATH = "tests/arena/test_results.csv"

@pytest.fixture
def engine():
    # Setup: Ensure clean slate
    if os.path.exists(LOCK_PATH): os.remove(LOCK_PATH)
    if os.path.exists(RESULTS_PATH): os.remove(RESULTS_PATH)
    
    return HardLockEngine(lock_path=LOCK_PATH, results_path=RESULTS_PATH)

def test_lock_match_persistence(engine):
    engine.lock_match(101)
    
    # Verify memory
    assert engine.is_locked(101)
    
    # Verify disk
    assert os.path.exists(LOCK_PATH)
    with open(LOCK_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["match_id"] == "101"

def test_double_lock_raises_error(engine):
    engine.lock_match(101)
    with pytest.raises(LockViolationError):
        engine.lock_match(101)

def test_future_kickoff_raises_error(engine):
    future_time = datetime.now(timezone.utc) + timedelta(hours=1)
    with pytest.raises(ValueError, match="kickoff is in the future"):
        engine.lock_match(102, kickoff_time=future_time)

def test_ingest_result_before_lock_raises_error(engine):
    with pytest.raises(MatchNotLockedError):
        engine.ingest_result(103, 2, 1, "Group Stage")

def test_ingest_result_success(engine):
    engine.lock_match(103)
    result = engine.ingest_result(103, 2, 1, "Group Stage")
    
    assert result.match_id == 103
    assert result.home_goals == 2
    
    # Verify persistence
    assert os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "r") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["match_id"] == "103"
        assert rows[0]["home_goals"] == "2"

def test_duplicate_result_raises_error(engine):
    engine.lock_match(104)
    engine.ingest_result(104, 1, 1, "Group Stage")
    with pytest.raises(ResultAlreadyExistsError):
        engine.ingest_result(104, 2, 2, "Group Stage")
