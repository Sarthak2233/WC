import pytest
from src.models.scoring import ContestScorer
from src.database import Prediction

def test_exact_match_score():
    scorer = ContestScorer()
    # E.g., predicted 2-1, actual 2-1 in Group Stage
    pred = Prediction(
        participant_name="test_user",
        match_id=1,
        predicted_home_score=2,
        predicted_away_score=1
    )
    score = scorer.calculate_score(pred, actual_home=2, actual_away=1, stage="Group Stage")
    assert score == 100

def test_correct_result_incorrect_score():
    scorer = ContestScorer()
    # predicted 2-0, actual 3-1 (Home win, but not exact)
    pred = Prediction(
        participant_name="test_user",
        match_id=1,
        predicted_home_score=2,
        predicted_away_score=0
    )
    score = scorer.calculate_score(pred, actual_home=3, actual_away=1, stage="Group Stage")
    assert score == 50

def test_incorrect_result():
    scorer = ContestScorer()
    # predicted 2-1 (Home win), actual 1-1 (Draw)
    pred = Prediction(
        participant_name="test_user",
        match_id=1,
        predicted_home_score=2,
        predicted_away_score=1
    )
    score = scorer.calculate_score(pred, actual_home=1, actual_away=1, stage="Group Stage")
    assert score == 0

def test_final_stage_multipliers():
    scorer = ContestScorer()
    pred = Prediction(
        participant_name="test_user",
        match_id=1,
        predicted_home_score=1,
        predicted_away_score=0
    )
    
    # R16 exact (1.5x)
    assert scorer.calculate_score(pred, 1, 0, "Round of 16") == 150
    # R16 result (1.5x)
    assert scorer.calculate_score(pred, 2, 1, "Round of 16") == 75
    
    # Final exact (4x)
    assert scorer.calculate_score(pred, 1, 0, "Final / 3rd Place") == 400
    # Final result (4x)
    assert scorer.calculate_score(pred, 2, 0, "Final / 3rd Place") == 200
