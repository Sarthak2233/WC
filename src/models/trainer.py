import os
import pandas as pd
import numpy as np
import logging
from typing import Any
from sklearn.linear_model import LinearRegression, BayesianRidge, PoissonRegressor, HuberRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from src.models.ensemble import StackingEnsemble

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Standard interface for training and predicting match outcomes.
    Uses HuberRegressor by default for robust regression to mitigate outlier (blowout) influence.
    Includes a normalization pipeline (RobustScaler -> StandardScaler).
    """
    def __init__(self, model=None, pipeline=None):
        self.model = model if model else HuberRegressor(max_iter=500)
        self.pipeline = pipeline if pipeline else Pipeline([
            ('robust', RobustScaler()),
            ('std', StandardScaler())
        ])
        self.is_fitted = False
        
    def reset(self):
        """Resets the trainer state."""
        self.is_fitted = False
        # Re-initialize the model to clear any internal state
        # For ensemble, this is trickier. For basic models, it's fine.
        # Let's assume for now re-fitting works if we handle pipeline correctly.
        # Wait, if I want to re-initialize model, I need to know the type.
        # Let's just create a new model instance.
        from sklearn.base import clone
        self.model = clone(self.model)
        
        self.pipeline = Pipeline([
            ('robust', RobustScaler()),
            ('std', StandardScaler())
        ])
        if hasattr(self, '_feature_columns'):
            del self._feature_columns

    def train(self, X: pd.DataFrame, y: Any):
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        # Filter out tournament_year and stage-related cols from training features if passed in X
        cols_to_drop = ["tournament_year", "stage_name"]
        X_numeric = X_numeric.drop(columns=[c for c in cols_to_drop if c in X_numeric.columns], errors="ignore")

        # Drop constant or near-constant columns (toggleable via FEATURE_DROP env var)
        feature_drop = os.environ.get('FEATURE_DROP', 'true').lower() in ('1', 'true', 'yes')
        const_cols = [c for c in X_numeric.columns if X_numeric[c].nunique() <= 1 or X_numeric[c].var() < 1e-8]
        if const_cols and feature_drop:
            logger.info(f"Dropping constant/near-constant columns: {const_cols[:10]}")
            X_numeric = X_numeric.drop(columns=const_cols, errors='ignore')
        elif const_cols and not feature_drop:
            logger.info(f"FEATURE_DROP disabled: keeping constant/near-constant columns: {const_cols[:10]}")

        # Check if features changed and we need to reset
        if self.is_fitted and hasattr(self, '_feature_columns'):
            if set(self._feature_columns) != set(X_numeric.columns):
                logging.getLogger(__name__).info("Features changed, resetting trainer.")
                self.reset()
        
        # Remember feature columns
        self._feature_columns = X_numeric.columns.tolist()

        # Fit and transform
        if not self.is_fitted:
            X_scaled = self.pipeline.fit_transform(X_numeric)
            self.is_fitted = True
        else:
            X_scaled = self.pipeline.transform(X_numeric)
            
        y_train = y['diff'] if isinstance(y, pd.DataFrame) and 'diff' in y.columns else y
        self.model.fit(X_scaled, y_train)

        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        if "tournament_year" in X_numeric.columns:
            X_numeric = X_numeric.drop(columns=["tournament_year"])
        
        # Ensure columns align with training
        if hasattr(self, '_feature_columns'):
            X_numeric = X_numeric.reindex(columns=self._feature_columns, fill_value=0)

        # Transform
        X_scaled = self.pipeline.transform(X_numeric)
        return self.model.predict(X_scaled)

class BaselineTrainer(ModelTrainer):
    """
    Trainer that only uses a specific subset of features (e.g., Elo only).
    """
    def __init__(self, feature_cols: list):
        super().__init__()
        self.feature_cols = feature_cols
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        # Ensure only numeric and existing features are used
        available_cols = [c for c in self.feature_cols if c in X.columns]
        super().train(X[available_cols], y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        available_cols = [c for c in self.feature_cols if c in X.columns]
        return super().predict(X[available_cols])

class OracleTrainer(ModelTrainer):
    """
    Trainer that uses the full StackingEnsemble.
    """
    def __init__(self):
        super().__init__(model=StackingEnsemble())
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        super().train(X, y)

class StageSpecializedTrainer(ModelTrainer):
    """
    Trainer that splits data into Group and Knockout stages and trains separate sub-models.
    """
    def __init__(self, group_trainer, knockout_trainer):
        super().__init__()
        self.group_trainer = group_trainer
        self.knockout_trainer = knockout_trainer
        
    def train(self, X: pd.DataFrame, y: Any):
        # Identify Group vs Knockout
        # Assume X has stage_name from matches
        if 'stage_name' not in X.columns:
            logger.warning("No stage_name in X, defaulting to group_trainer for all.")
            self.group_trainer.train(X, y)
            return

        is_knockout = X['stage_name'].str.lower().apply(lambda x: 'group' not in str(x).lower())
        
        X_group = X[~is_knockout]
        y_group = y[~is_knockout] if not isinstance(y, pd.Series) else y[~is_knockout]
        
        X_knockout = X[is_knockout]
        y_knockout = y[is_knockout] if not isinstance(y, pd.Series) else y[is_knockout]
        
        if not X_group.empty:
            self.group_trainer.train(X_group, y_group)
        if not X_knockout.empty:
            self.knockout_trainer.train(X_knockout, y_knockout)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if 'stage_name' not in X.columns:
            return self.group_trainer.predict(X)
            
        is_knockout = X['stage_name'].str.lower().apply(lambda x: 'group' not in str(x).lower())
        
        preds = np.zeros(len(X))
        
        if not X[~is_knockout].empty:
            preds[~is_knockout] = self.group_trainer.predict(X[~is_knockout])
        if not X[is_knockout].empty:
            preds[is_knockout] = self.knockout_trainer.predict(X[is_knockout])
            
        return preds

class BayesianHierarchicalTrainer(ModelTrainer):
    """
    Trainer that uses Bayesian Ridge Regression and accounts for 
    hierarchical features like match stage via one-hot encoding.
    """
    def __init__(self):
        super().__init__(model=BayesianRidge())
        
    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = X.copy()
        
        # Handle 'stage_name' if present
        if "stage_name" in X_processed.columns:
            X_processed["is_knockout"] = X_processed["stage_name"].apply(
                lambda x: 0 if "group" in str(x).lower() else 1
            )
            X_processed = X_processed.drop(columns=["stage_name"])
            
        X_numeric = X_processed.select_dtypes(include=[np.number]).fillna(0)
        if "tournament_year" in X_numeric.columns:
            X_numeric = X_numeric.drop(columns=["tournament_year"])
            
        return X_numeric

    def train(self, X: pd.DataFrame, y: pd.Series):
        X_clean = self._preprocess(X)
        super().train(X_clean, y)
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = self._preprocess(X)
        return super().predict(X_clean)

class PoissonTrainer(ModelTrainer):
    """
    Trainer that uses Poisson Regression for goal prediction (bounded to >= 0).
    """
    def __init__(self):
        # Reduce regularization to avoid zeroed coefficients when informative features exist
        # Reduce regularization and increase iterations to avoid zeroing coefficients
        super().__init__(model=PoissonRegressor(alpha=1e-8, max_iter=2000, tol=1e-7))
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        # Ensure target is non-negative for Poisson
        y_clipped = y.clip(lower=0)
        # First attempt: Poisson
        super().train(X, y_clipped)

        # If Poisson coefficients are all effectively zero, fallback to a dense regressor
        try:
            coef = getattr(self.model, 'coef_', None)
            if coef is not None and (abs(coef).sum() == 0):
                logger.warning("PoissonTrainer: all coefficients zero — falling back to BayesianRidge.")
                from sklearn.linear_model import BayesianRidge
                self.model = BayesianRidge()
                # Re-train with same preprocessing
                super().train(X, y_clipped)
                self._fallback = 'BayesianRidge'
        except Exception:
            logger.exception("Error while checking Poisson coefficients.")
            # If something goes wrong, leave model as-is
            pass
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Bounding output to [0, 8]
        preds = super().predict(X)
        return np.clip(preds, 0, 8)

class DoublePoissonTrainer(ModelTrainer):
    """
    Wrapper that trains two Poisson models (Home/Away) and returns predicted difference.
    Supports a unified interface where X contains 'home_' and 'away_' prefixes.
    """
    def __init__(self):
        super().__init__()
        self.home_model = PoissonTrainer()
        self.away_model = PoissonTrainer()
        
    def train(self, X: pd.DataFrame, y: Any):
        # Determine y_home and y_away
        if isinstance(y, pd.DataFrame) and 'home_goals' in y.columns:
            y_home, y_away = y['home_goals'], y['away_goals']
        elif hasattr(y, 'y_home'): # Custom attribute if passed
            y_home, y_away = y.y_home, y.y_away
        else:
            # If y is just a difference, we can't train Poisson properly
            # but for benchmarking we might have pre-split them or we skip.
            logger.warning("DoublePoissonTrainer needs absolute goals for training.")
            return

        # Split X into home and away features
        home_cols = [c for c in X.columns if c.startswith("home_")]
        away_cols = [c for c in X.columns if c.startswith("away_")]
        
        X_home = X[home_cols].rename(columns=lambda x: x.replace("home_", ""))
        X_away = X[away_cols].rename(columns=lambda x: x.replace("away_", ""))
        
        self.home_model.train(X_home, y_home)
        self.away_model.train(X_away, y_away)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        home_cols = [c for c in X.columns if c.startswith("home_")]
        away_cols = [c for c in X.columns if c.startswith("away_")]
        
        X_home = X[home_cols].rename(columns=lambda x: x.replace("home_", ""))
        X_away = X[away_cols].rename(columns=lambda x: x.replace("away_", ""))
        
        h_preds = self.home_model.predict(X_home)
        a_preds = self.away_model.predict(X_away)
        return h_preds - a_preds

    def predict_diff(self, X_home: pd.DataFrame, X_away: pd.DataFrame) -> np.ndarray:
        h_preds = self.home_model.predict(X_home)
        a_preds = self.away_model.predict(X_away)
        return h_preds - a_preds

class ConsensusOracle(ModelTrainer):
    """
    Consensus Oracle that uses Meta-Stacking to combine:
    1. Double Poisson (Goal Difference)
    2. Bayesian Hierarchical
    3. Stacking Ensemble (XGB + LGB)
    
    Supports a unified 'Master Matrix' with home_, away_, and diff_ prefixes.
    """
    def __init__(self):
        from sklearn.linear_model import Ridge
        super().__init__(model=Ridge(alpha=1.0))
        self.poisson = DoublePoissonTrainer()
        self.bayesian = BayesianHierarchicalTrainer()
        self.ensemble = OracleTrainer()
        self._is_trained = False

    def train(self, X: pd.DataFrame, y: Any):
        """Standard train interface for Benchmarking."""
        # 1. Prepare target
        if isinstance(y, pd.DataFrame) and 'diff' in y.columns:
            y_diff = y['diff']
            y_master = y
        else:
            y_diff = y
            y_master = y # Might be just a series

        # 2. Extract feature sets
        home_cols = [c for c in X.columns if c.startswith("home_")]
        away_cols = [c for c in X.columns if c.startswith("away_")]
        diff_cols = [c for c in X.columns if c.startswith("diff_")]
        
        X_home = X[home_cols].rename(columns=lambda x: x.replace("home_", ""))
        X_away = X[away_cols].rename(columns=lambda x: x.replace("away_", ""))
        X_diff = X[diff_cols].rename(columns=lambda x: x.replace("diff_", ""))
        
        # 3. Train Sub-models
        logger.info("Consensus: Training sub-models...")
        self.poisson.train(X, y_master) # poisson knows how to split X
        self.bayesian.train(X_diff, y_diff)
        self.ensemble.train(X_diff, y_diff)
        
        # 4. Generate Meta-features
        p_preds = self.poisson.predict(X)
        b_preds = self.bayesian.predict(X_diff)
        e_preds = self.ensemble.predict(X_diff)
        
        meta_X = pd.DataFrame({
            "poisson": p_preds,
            "bayesian": b_preds,
            "ensemble": e_preds
        })
        
        # 5. Train Meta-learner
        logger.info("Consensus: Training meta-learner...")
        self.model.fit(meta_X, y_diff)
        self._is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Consensus Oracle not trained.")
            
        diff_cols = [c for c in X.columns if c.startswith("diff_")]
        X_diff = X[diff_cols].rename(columns=lambda x: x.replace("diff_", ""))
        
        p_preds = self.poisson.predict(X)
        b_preds = self.bayesian.predict(X_diff)
        e_preds = self.ensemble.predict(X_diff)
        
        meta_X = pd.DataFrame({
            "poisson": p_preds,
            "bayesian": b_preds,
            "ensemble": e_preds
        })
        
        return self.model.predict(meta_X)

    def predict_match(self, X_home: pd.DataFrame, X_away: pd.DataFrame) -> np.ndarray:
        """Explicit predict for live 2026 matches."""
        X_diff = X_home.reset_index(drop=True) - X_away.reset_index(drop=True)
        X_diff.columns = [f"diff_{c}" for c in X_diff.columns]
        
        # Wrap for sub-model calls
        X_master = pd.concat([
            X_diff, 
            X_home.reset_index(drop=True).add_prefix("home_"),
            X_away.reset_index(drop=True).add_prefix("away_")
        ], axis=1)
        
        return self.predict(X_master)


