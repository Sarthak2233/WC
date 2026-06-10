import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

class StackingEnsemble:
    """
    Super-model that stacks predictions from multiple base models (XGBoost, LightGBM)
    using a meta-learner (Ridge Regression).
    """
    
    def __init__(self):
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.meta_model = Ridge(alpha=1.0)
        self._is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the base models and the meta-model on the same data.
        In a robust implementation, this would use K-Fold cross validation 
        for the meta-model training to avoid overfitting.
        """
        logger.info("Training base models for ensemble...")
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        
        # Get base predictions
        xgb_preds = self.xgb_model.predict(X)
        lgb_preds = self.lgb_model.predict(X)
        
        # Stack predictions as features for the meta-model
        meta_X = pd.DataFrame({
            "xgb": xgb_preds,
            "lgb": lgb_preds
        })
        
        logger.info("Training meta model...")
        self.meta_model.fit(meta_X, y)
        self._is_trained = True
        logger.info("Ensemble training complete.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts using the stacked ensemble.
        """
        if not self._is_trained:
            raise ValueError("Ensemble has not been trained yet.")
            
        xgb_preds = self.xgb_model.predict(X)
        lgb_preds = self.lgb_model.predict(X)
        
        meta_X = pd.DataFrame({
            "xgb": xgb_preds,
            "lgb": lgb_preds
        })
        
        return self.meta_model.predict(meta_X)
