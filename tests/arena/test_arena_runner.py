import pytest
import os
import csv
from datetime import datetime, timezone
from src.arena.hard_lock_engine import HardLockEngine
from src.arena.arena_runner import ArenaRunner
from src.database import Prediction, append_prediction

# Temp files
PRED_PATH = "tests/arena/test_preds.csv"
LOCK_PATH = "tests/arena/test_lock_runner.csv"
RESULTS_PATH = "tests/arena/test_results_runner.csv"
SCORES_PATH = "tests/arena/test_scores_runner.csv"

@pytest.fixture
def runner():
    # Setup: Clean slate
    for path in [PRED_PATH, LOCK_PATH, RESULTS_PATH, SCORES_PATH]:
        if os.path.exists(path): os.remove(path)
        
    engine = HardLockEngine(lock_path=LOCK_PATH, results_path=RESULTS_PATH)
    return ArenaRunner(lock_engine=engine, scores_path=SCORES_PATH, predictions_path=PRED_PATH)

def test_process_result_pipeline(runner):
    # 1. Setup Data
    match_id = 201
    # Add a prediction
    pred = Prediction(
        participant_name="UserA",
        match_id=match_id,
        predicted_home_score=2,
        predicted_away_score=1,
        stage="Group Stage"
    )
    append_prediction(pred, path=PRED_PATH)
    
    # 2. Lock
    runner._lock_engine.lock_match(match_id)
    
    # 3. Process
    leaderboard = runner.process_result(match_id, 2, 1, "Group Stage")
    
    # 4. Assert
    # Pipeline should have scored the user
    assert len(leaderboard) == 1
    assert leaderboard[0]["name"] == "UserA"
    assert leaderboard[0]["total_points"] == 100 # Exact match
    
    # Verify persistence
    assert os.path.exists(SCORES_PATH)
    with open(SCORES_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["points"] == "100"
