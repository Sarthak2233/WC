import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Evaluates model performance using LOTO (Leave-One-Tournament-Out) cross-validation.
    """
    
    def leave_one_tournament_out(self, trainer, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Performs LOTO CV.
        Expects X to have a 'tournament_year' column.
        """
        if 'tournament_year' not in X.columns:
            raise ValueError("Feature matrix must contain 'tournament_year' for LOTO CV.")
            
        years = X['tournament_year'].unique()
        
        per_tournament_rmse = {}
        all_preds = []
        all_actuals = []
        
        for year in years:
            train_idx = X['tournament_year'] != year
            test_idx = X['tournament_year'] == year
            
            X_train, y_train = X[train_idx].drop(columns=['tournament_year']), y[train_idx]
            X_test, y_test = X[test_idx].drop(columns=['tournament_year']), y[test_idx]
            
            if X_train.empty or X_test.empty:
                continue
                
            # Train model
            trainer.train(X_train, y_train)
            
            # Predict
            preds = trainer.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            per_tournament_rmse[year] = rmse
            
            all_preds.extend(preds)
            all_actuals.extend(y_test)
            
        overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_preds)) if all_actuals else 0.0
        
        return {
            "overall_rmse": overall_rmse,
            "per_tournament_rmse": per_tournament_rmse
        }
