import pytest
import pandas as pd
from src.models.simulator import TournamentSimulator

class MockModel:
    def predict(self, X):
        return [2.0] * len(X) # always predicts a draw basically if score diff is used
        
def test_simulate_match():
    simulator = TournamentSimulator(model=MockModel())
    
    # Normally it predicts goal difference, say. 
    # For testing, we mock random.gauss inside simulate_match.
    import random
    random.seed(42)
    
    team_a = {"country_code": "BRA", "ppi": 0.5}
    team_b = {"country_code": "ARG", "ppi": 0.6}
    
    home_goals, away_goals = simulator.simulate_match(team_a, team_b, is_knockout=False)
    
    assert isinstance(home_goals, int)
    assert isinstance(away_goals, int)
    assert home_goals >= 0
    assert away_goals >= 0

def test_monte_carlo():
    simulator = TournamentSimulator(model=MockModel())
    
    team_a = {"country_code": "BRA", "ppi": 0.5}
    team_b = {"country_code": "ARG", "ppi": 0.6}
    
    results = simulator.monte_carlo_match(team_a, team_b, iterations=100)
    
    assert "home_win_prob" in results
    assert "away_win_prob" in results
    assert "draw_prob" in results
    
    assert abs(results["home_win_prob"] + results["away_win_prob"] + results["draw_prob"] - 1.0) < 0.001
