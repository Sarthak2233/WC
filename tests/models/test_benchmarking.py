import pytest
import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from src.models.evaluator import Evaluator
from src.models.benchmarking import BenchmarkingEngine

def test_benchmarking_engine_comparison():
    evaluator = Evaluator()
    benchmarker = BenchmarkingEngine(evaluator)
    
    X = pd.DataFrame({
        "tournament_year": [2014, 2014, 2018, 2018, 2022, 2022],
        "feature1": [1, 2, 3, 4, 5, 6]
    })
    # y is basically feature1 * 2 for model1, but random for others
    y = pd.Series([2, 4, 6, 8, 10, 12])
    
    # Model 1: Good model
    trainer1 = ModelTrainer()
    
    # Model 2: Constant predictor (baseline)
    class ConstantTrainer:
        def train(self, X, y): pass
        def predict(self, X): return np.zeros(len(X))
    
    trainer2 = ConstantTrainer()
    
    trainers = {
        "LinearRegression": trainer1,
        "ConstantZero": trainer2
    }
    
    report = benchmarker.run_benchmark(trainers, X, y)
    
    assert "LinearRegression" in report.index
    assert "ConstantZero" in report.index
    assert report.loc["LinearRegression", "overall_rmse"] < report.loc["ConstantZero", "overall_rmse"]
