import pandas as pd
import numpy as np
import logging
from src.features.csv_oracle import CSVFeatureOracle
from src.models.ensemble import StackingEnsemble
from sklearn.metrics import accuracy_score, mean_absolute_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AblationStudy:
    def __init__(self, processed_dir="data/processed"):
        self.engine = CSVFeatureOracle(processed_dir)
        
    def run_study(self, test_year=2022):
        logger.info(f"--- Phase 4: Ablation Study (Test Year: {test_year}) ---")
        
        matches = pd.read_csv(os.path.join(self.engine.processed_dir, "matches.csv"))
        train_matches = matches[matches["tournament_year"] < test_year]
        test_matches = matches[matches["tournament_year"] == test_year]
        
        def get_xy(match_set):
            x_rows, y_vals = [], []
            for _, m in match_set.iterrows():
                year = int(m["tournament_year"])
                f1 = self.engine.get_team_features(m["home_team"], year)
                f2 = self.engine.get_team_features(m["away_team"], year)
                diff = (f1 - f2).add_prefix("diff_")
                x_rows.append(diff)
                y_vals.append(m["home_team_score"] - m["away_team_score"])
            return pd.DataFrame(x_rows), pd.Series(y_vals)

        X_train_raw, y_train = get_xy(train_matches)
        X_test_raw, y_test = get_xy(test_matches)
        
        X_train = X_train_raw.fillna(0)
        X_test = X_test_raw.fillna(0)
        
        feature_sets = {
            "Elo Only": [c for c in X_train.columns if "elo" in c],
            "Elo + Tournament": [c for c in X_train.columns if "elo" in c or "is_host" in c or "is_defending" in c or "legacy" in c],
            "Elo + Tournament + Politics": [c for c in X_train.columns if "elo" in c or "is_host" in c or "is_defending" in c or "legacy" in c or "political" in c or "gdp" in c or "conflict" in c],
            "Elo + Tournament + Culture": [c for c in X_train.columns if "elo" in c or "is_host" in c or "is_defending" in c or "legacy" in c or "pdi" in c or "idv" in c or "mas" in c or "uai" in c or "happiness" in c],
            "Full Oracle": X_train.columns.tolist()
        }
        
        results = []
        for name, cols in feature_sets.items():
            logger.info(f"Evaluating: {name}")
            model = StackingEnsemble()
            model.train(X_train[cols], y_train)
            preds = model.predict(X_test[cols])
            
            mae = mean_absolute_error(y_test, preds)
            def to_res(s):
                if s > 0: return 1
                if s < 0: return -1
                return 0
            acc = accuracy_score(y_test.apply(to_res), pd.Series(preds).apply(to_res))
            
            results.append({"Model": name, "Accuracy": acc, "MAE": mae})
            
        print("\n--- Ablation Results ---")
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    import os
    study = AblationStudy()
    study.run_study(2022)
    study.run_study(2018)
