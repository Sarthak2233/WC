import pytest
from datetime import datetime
from src.models.leaderboard import LeaderboardSystem

def test_leaderboard_sorting():
    lb = LeaderboardSystem()
    
    # User A: 500 pts, 2 exact, 100 KO, time 1
    # User B: 500 pts, 3 exact, 50 KO, time 2
    # User C: 500 pts, 2 exact, 150 KO, time 3
    # User D: 600 pts
    
    participants = [
        {"name": "A", "total_points": 500, "exact_hits": 2, "ko_points": 100, "first_pred_time": datetime(2026, 1, 1, 10, 0)},
        {"name": "B", "total_points": 500, "exact_hits": 3, "ko_points": 50, "first_pred_time": datetime(2026, 1, 1, 11, 0)},
        {"name": "C", "total_points": 500, "exact_hits": 2, "ko_points": 150, "first_pred_time": datetime(2026, 1, 1, 12, 0)},
        {"name": "D", "total_points": 600, "exact_hits": 1, "ko_points": 0, "first_pred_time": datetime(2026, 1, 1, 9, 0)}
    ]
    
    ranked = lb.rank_participants(participants)
    
    # 1. D (600 pts)
    assert ranked[0]["name"] == "D"
    
    # 2. B (500 pts, 3 exact)
    assert ranked[1]["name"] == "B"
    
    # 3. C (500 pts, 2 exact, 150 KO)
    assert ranked[2]["name"] == "C"
    
    # 4. A (500 pts, 2 exact, 100 KO)
    assert ranked[3]["name"] == "A"

def test_leaderboard_timestamp_tiebreak():
    lb = LeaderboardSystem()
    
    participants = [
        {"name": "Late", "total_points": 500, "exact_hits": 2, "ko_points": 100, "first_pred_time": datetime(2026, 1, 1, 11, 0)},
        {"name": "Early", "total_points": 500, "exact_hits": 2, "ko_points": 100, "first_pred_time": datetime(2026, 1, 1, 10, 0)}
    ]
    
    ranked = lb.rank_participants(participants)
    assert ranked[0]["name"] == "Early"
    assert ranked[1]["name"] == "Late"
