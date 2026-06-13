import pandas as pd
import numpy as np
import pickle
import os
import logging
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from src.features.csv_oracle import CSVFeatureOracle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    """
    Handles evaluation of models using various strategies like 
    Leave-One-Tournament-Out (LOTO) cross-validation.
    """
    
    def expanding_window_evaluation(self, trainer, X: pd.DataFrame, y: Any):
        """
        Performs tournament-by-tournament validation: 
        Train on all years < year, test on year.
        """
        if 'tournament_year' not in X.columns:
            raise ValueError("X must contain 'tournament_year' column.")
            
        years = sorted(X['tournament_year'].unique())
        # Need at least 2 tournaments to do expanding window
        if len(years) < 2:
            return self.leave_one_tournament_out(trainer, X, y)
            
        all_preds = []
        all_actuals = []
        
        def to_result(score):
            try:
                val = float(score)
            except:
                return 0
            if val > 0: return 1
            if val < 0: return -1
            return 0

        # Start testing from the 3rd tournament to have enough training data
        for i in range(2, len(years)):
            test_year = years[i]
            train_years = years[:i]
            
            X_train = X[X['tournament_year'].isin(train_years)]
            y_train = y[X['tournament_year'].isin(train_years)]
            X_test = X[X['tournament_year'] == test_year]
            y_test = y[X['tournament_year'] == test_year]
            
            # Save splits
            split_dir = f"models/data_splits/fold_{test_year}"
            os.makedirs(split_dir, exist_ok=True)
            X_train.to_csv(f"{split_dir}/X_train.csv", index=False)
            X_test.to_csv(f"{split_dir}/X_test.csv", index=False)
            if isinstance(y_train, pd.DataFrame):
                y_train.to_csv(f"{split_dir}/y_train.csv", index=False)
                y_test.to_csv(f"{split_dir}/y_test.csv", index=False)
            else:
                y_train.to_csv(f"{split_dir}/y_train.csv", index=False, header=['y'])
                y_test.to_csv(f"{split_dir}/y_test.csv", index=False, header=['y'])

            y_test_diff = y_test['diff'] if isinstance(y_test, pd.DataFrame) else y_test

            trainer.train(X_train, y_train)
            preds = trainer.predict(X_test)
            
            all_preds.extend(preds)
            all_actuals.extend(y_test_diff)
            
        all_actuals_series = pd.Series(all_actuals)
        all_preds_series = pd.Series(all_preds)
        
        overall_rmse = np.sqrt(mean_squared_error(all_actuals_series, all_preds_series))
        overall_mae = mean_absolute_error(all_actuals_series, all_preds_series)
        overall_acc = accuracy_score(all_actuals_series.apply(to_result), 
                                    all_preds_series.apply(to_result))
        
        return {
            "overall_rmse": overall_rmse,
            "overall_mae": overall_mae,
            "overall_accuracy": overall_acc
        }
    def leave_one_tournament_out(self, trainer, X: pd.DataFrame, y: Any):
        """
        Performs Leave-One-Tournament-Out cross-validation.
        Assumes X has a 'tournament_year' column.
        """
        if 'tournament_year' not in X.columns:
            raise ValueError("X must contain 'tournament_year' column for LOTO.")
            
        years = X['tournament_year'].unique()
        per_tournament_rmse = {}
        all_preds = []
        all_actuals = []
        
        def to_result(score):
            try:
                val = float(score)
            except:
                return 0
            if val > 0: return 1
            if val < 0: return -1
            return 0

        for year in years:
            # Split
            X_train = X[X['tournament_year'] != year]
            y_train = y[X['tournament_year'] != year]
            X_test = X[X['tournament_year'] == year]
            y_test = y[X['tournament_year'] == year]
            
            # Use 'diff' column for metrics if y is a DataFrame
            y_test_diff = y_test['diff'] if isinstance(y_test, pd.DataFrame) else y_test

            # Train and Predict
            trainer.train(X_train, y_train)
            preds = trainer.predict(X_test)
            
            # Record results
            rmse = np.sqrt(mean_squared_error(y_test_diff, preds))
            
            # Match result accuracy
            actual_results = y_test_diff.apply(to_result)
            pred_results = pd.Series(preds).apply(to_result)
            acc = accuracy_score(actual_results, pred_results)

            per_tournament_rmse[int(year)] = rmse
            all_preds.extend(preds)
            all_actuals.extend(y_test_diff)
            
        all_actuals_series = pd.Series(all_actuals)
        all_preds_series = pd.Series(all_preds)
        
        overall_rmse = np.sqrt(mean_squared_error(all_actuals_series, all_preds_series))
        overall_mae = mean_absolute_error(all_actuals_series, all_preds_series)
        overall_acc = accuracy_score(all_actuals_series.apply(to_result), 
                                    all_preds_series.apply(to_result))
        
        return {
            "overall_rmse": overall_rmse,
            "overall_mae": overall_mae,
            "overall_accuracy": overall_acc,
            "per_tournament_rmse": per_tournament_rmse
        }


def evaluate_model():
    """
    Script-style evaluation for the current best Oracle model.
    """
    model_path = "models/oracle_v1.pkl"
    if not os.path.exists(model_path):
        logger.error(f"Oracle model not found at {model_path}.")
        return
        
    with open(model_path, "rb") as f:
        oracle = pickle.load(f)
        
    engine = CSVFeatureOracle("data/processed")
    X, y_actual, _ = engine.build_training_set()
    
    # Generate predictions
    y_pred = oracle.predict(X)
    
    # 1. Regression Metrics (Goal Difference)
    mae = mean_absolute_error(y_actual, y_pred)
    
    # 2. Classification Metrics (Win/Loss/Draw)
    def to_result(score):
        if score > 0: return 1 # Win
        if score < 0: return -1 # Loss
        return 0 # Draw
        
    actual_results = y_actual.apply(to_result)
    pred_results = pd.Series(y_pred).apply(to_result)
    
    accuracy = accuracy_score(actual_results, pred_results)
    
    logger.info("--- Model Performance Statistics ---")
    logger.info(f"Mean Absolute Error (Goal Diff): {mae:.4f}")
    logger.info(f"Match Result Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model()
