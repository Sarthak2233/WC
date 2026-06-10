import pandas as pd
import numpy as np
import xgboost as xgb
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Trains the ML model (XGBoost) for World Cup predictions
    based on the 11-layer feature matrix.
    """
    
    def __init__(self, model_path: str = "models/xgboost_oracle.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the XGBoost regressor.
        """
        logger.info(f"Training XGBoost model on {len(X)} samples with {len(X.columns)} features.")
        
        # In a real scenario we'd do hyperparameter tuning, cross validation, etc.
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='reg:squarederror',
            random_state=42
        )
        
        self.model.fit(X, y)
        logger.info("Model training complete.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts outcomes for a given feature matrix.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
        
    def save(self) -> None:
        """Saves the model to disk."""
        if self.model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self) -> None:
        """Loads the model from disk."""
        if self.model_path.exists():
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
