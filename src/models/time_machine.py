import pandas as pd
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import accuracy_score, mean_absolute_error
from src.features.csv_oracle import CSVFeatureOracle
from src.models.ensemble import StackingEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeMachineEvaluator:
    """
    Skeptical auditor that performs walk-forward validation 
    to prove/falsify the model's predictive power.
    """
    
    def __init__(self, processed_dir="data/processed"):
        self.engine = CSVFeatureOracle(processed_dir)
        
    def run_audit(self, test_year=2022):
        """
        Trains on data strictly BEFORE test_year and predicts test_year.
        """
        logger.info(f"--- Phase 0: Time Machine Audit (Test Year: {test_year}) ---")
        
        # 1. Build the full historical dataset
        # We need to reach into the engine and get the raw match data
        matches = pd.read_csv(os.path.join(self.engine.processed_dir, "matches.csv"))
        
        # 2. Split matches by time
        train_matches = matches[matches["tournament_year"] < test_year]
        test_matches = matches[matches["tournament_year"] == test_year]
        
        if test_matches.empty:
            logger.error(f"No matches found for year {test_year}")
            return
            
        logger.info(f"Training on {len(train_matches)} matches before {test_year}")
        logger.info(f"Testing on {len(test_matches)} matches in {test_year}")
        
        # 3. Extract features for Train and Test
        # Helper to build X, y for a specific match set
        def get_xy(match_set):
            x_rows = []
            y_vals = []
            for _, m in match_set.iterrows():
                year = int(m["tournament_year"])
                f1 = self.engine.get_team_features(m["home_team"], year)
                f2 = self.engine.get_team_features(m["away_team"], year)
                diff = (f1 - f2).add_prefix("diff_")
                x_rows.append(diff)
                y_vals.append(m["home_team_score"] - m["away_team_score"])
            return pd.DataFrame(x_rows), pd.Series(y_vals)

        X_train, y_train = get_xy(train_matches)
        X_test, y_test = get_xy(test_matches)
        
        # 4. Baselines
        # Elo-Only Baseline
        elo_col = "diff_elo_elo"
        X_train_elo = X_train[[elo_col]].fillna(0)
        X_test_elo = X_test[[elo_col]].fillna(0)
        
        from sklearn.linear_model import LinearRegression
        elo_model = LinearRegression()
        elo_model.fit(X_train_elo, y_train)
        elo_preds = elo_model.predict(X_test_elo)
        
        # 5. Full Oracle
        oracle = StackingEnsemble()
        oracle.train(X_train.fillna(0), y_train)
        oracle_preds = oracle.predict(X_test.fillna(0))
        
        # 6. Metrics Calculation
        def evaluate(y_true, y_pred, label):
            mae = mean_absolute_error(y_true, y_pred)
            
            def to_res(s):
                if s > 0: return 1
                if s < 0: return -1
                return 0
            
            acc = accuracy_score(y_true.apply(to_res), pd.Series(y_pred).apply(to_res))
            logger.info(f"[{label}] Accuracy: {acc:.4f}, MAE: {mae:.4f}")
            return acc, mae

        logger.info("--- Audit Results ---")
        acc_elo, mae_elo = evaluate(y_test, elo_preds, "Elo-Only Baseline")
        acc_ora, mae_ora = evaluate(y_test, oracle_preds, "Full Oracle Model")
        
        diff_acc = acc_ora - acc_elo
        diff_mae = (mae_elo - mae_ora) / mae_elo if mae_elo != 0 else 0
        
        logger.info(f"Improvement - Accuracy: {diff_acc:+.4f}, MAE Reduction: {diff_mae:.2%}")
        
        if diff_acc < 0.03 and diff_mae < 0.05:
            logger.warning("FALSIFIED: Oracle failed to meaningfully outperform Elo baseline.")
        else:
            logger.info("PASSED: Oracle shows meaningful improvement.")

if __name__ == "__main__":
    auditor = TimeMachineEvaluator()
    auditor.run_audit(2022)
    auditor.run_audit(2018)
