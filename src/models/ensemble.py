import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)

from sklearn.base import BaseEstimator, RegressorMixin

class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Super-model that stacks predictions from multiple base models (XGBoost, LightGBM)
    using a meta-learner (Ridge Regression).
    """
    
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42
        )
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42
        )
        self.meta_model = Ridge(alpha=1.0)
        self._is_trained = False
        
    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Re-initialize models with new params
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42
        )
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate, random_state=42
        )
        return self
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Backwards-compatible alias for other trainer interfaces
        self.train = self.fit

        """
        Trains the base models and the meta-model on the same data.
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
