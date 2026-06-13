import pandas as pd
import logging
from src.features.csv_oracle import CSVFeatureOracle
from src.models.evaluator import Evaluator
from src.models.trainer import BaselineTrainer, OracleTrainer, BayesianHierarchicalTrainer
from src.models.benchmarking import BenchmarkingEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmarking_suite():
    engine = CSVFeatureOracle("data/processed")
    X_home, y_home, X_away, y_away, filtered_matches = engine.build_absolute_training_set()

    # 1. Build Master Matrix
    # Reset indices to ensure alignment
    X_home = X_home.reset_index(drop=True)
    X_away = X_away.reset_index(drop=True)
    y_home = y_home.reset_index(drop=True)
    y_away = y_away.reset_index(drop=True)

    # Difference Matrix
    X_diff = (X_home - X_away)
    X_diff.columns = [f"diff_{c}" for c in X_diff.columns]
    
    # Master Matrix (prefixed)
    X = pd.concat([
        X_diff,
        X_home.add_prefix("home_"),
        X_away.add_prefix("away_")
    ], axis=1)

    X["tournament_year"] = filtered_matches["tournament_year"].values
    X["stage_name"] = filtered_matches["stage_name"].values
    
    # Target Dataframe for trainers that need more than just diff
    y_master = pd.DataFrame({
        "diff": y_home - y_away,
        "home_goals": y_home,
        "away_goals": y_away
    })

    evaluator = Evaluator()
    benchmarker = BenchmarkingEngine(evaluator)
    
    from src.models.trainer import DoublePoissonTrainer, ConsensusOracle, StageSpecializedTrainer, OracleTrainer

    trainers = {
        "Elo-Only": BaselineTrainer(feature_cols=["diff_elo_elo"]),
        "Politics-Only": BaselineTrainer(feature_cols=["diff_political_stability_lag1", "diff_gdp_per_capita_lag1", "diff_conflict_intensity_lag1"]),
        "Culture-Only": BaselineTrainer(feature_cols=["diff_pdi", "diff_idv", "diff_mas", "diff_uai", "diff_happiness_score_lag1"]),
        "Double-Poisson": DoublePoissonTrainer(),
        "Bayesian-Hierarchical": BayesianHierarchicalTrainer(),
        "Stacking-Ensemble": StageSpecializedTrainer(group_trainer=OracleTrainer(), knockout_trainer=OracleTrainer()),
        "CONSENSUS-ORACLE": ConsensusOracle()
    }
    # Standard trainers only use X_diff, but Consensus uses full X
    # We can pass y_master to all; standard ones will just use the series if handled.
    # However, to be safe, we let BenchmarkingEngine handle y choice if needed.
    
    report = benchmarker.run_benchmark(trainers, X, y_master)

    
    print("\n--- GLOBAL BENCHMARK REPORT ---")
    print(report.to_string())
    
    # Save report
    report.to_csv("models/benchmark_report.csv")
    logger.info("Benchmark report saved to models/benchmark_report.csv")

if __name__ == "__main__":
    run_benchmarking_suite()
