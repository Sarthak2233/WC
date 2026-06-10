import pytest
import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer

def test_train_model():
    trainer = ModelTrainer()
    
    # Mock feature matrix
    X = pd.DataFrame({
        "ppi": [0.2, 0.8, 0.5, 0.1],
        "adversity_mean": [5.0, 8.0, 3.0, 1.0],
        "psyche_score": [0.6, 0.4, 0.5, 0.8],
        "is_host": [1.0, 0.0, 0.0, 0.0]
    })
    
    # Mock target (e.g., points earned or goal difference)
    y = pd.Series([9, 0, 3, 4])
    
    trainer.train(X, y)
    
    assert trainer.model is not None
    
    # Predict
    preds = trainer.predict(X)
    assert len(preds) == 4
    assert isinstance(preds[0], (float, np.floating))
