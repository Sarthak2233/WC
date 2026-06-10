import pytest
import pandas as pd
from src.models.evaluator import Evaluator
from src.models.trainer import ModelTrainer

def test_loto_cross_validation():
    evaluator = Evaluator()
    trainer = ModelTrainer()
    
    X = pd.DataFrame({
        "tournament_year": [2014, 2014, 2018, 2018, 2022, 2022],
        "feature1": [1, 2, 3, 4, 5, 6]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0])
    
    results = evaluator.leave_one_tournament_out(trainer, X, y)
    
    assert "overall_rmse" in results
    assert "per_tournament_rmse" in results
    assert len(results["per_tournament_rmse"]) == 3
    assert 2014 in results["per_tournament_rmse"]
