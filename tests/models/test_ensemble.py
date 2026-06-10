import pytest
import pandas as pd
import numpy as np
from src.models.ensemble import StackingEnsemble

def test_ensemble_train_predict():
    ensemble = StackingEnsemble()
    
    # Mock data
    X = pd.DataFrame({
        "ppi": [0.2, 0.8, 0.5, 0.1, 0.9, 0.3],
        "adversity_mean": [5.0, 8.0, 3.0, 1.0, 9.0, 2.0],
        "psyche_score": [0.6, 0.4, 0.5, 0.8, 0.2, 0.9]
    })
    y = pd.Series([9, 0, 3, 4, 1, 7])
    
    # Train
    ensemble.train(X, y)
    
    # Predict
    preds = ensemble.predict(X)
    
    assert len(preds) == 6
    assert isinstance(preds[0], (float, np.floating))
