import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BenchmarkingEngine:
    """
    Orchestrates comparison between multiple prediction models.
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        
    def run_benchmark(self, trainers: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Runs tournament-by-tournament expanding window evaluation for each trainer.
        """
        results = []
        for name, trainer in trainers.items():
            logger.info(f"Benchmarking model: {name}")
            try:
                metrics = self.evaluator.expanding_window_evaluation(trainer, X, y)
                results.append({
                    "model": name,
                    "overall_rmse": metrics["overall_rmse"],
                    "overall_mae": metrics["overall_mae"],
                    "overall_accuracy": metrics["overall_accuracy"]
                })
            except Exception as e:
                logger.error(f"Failed to benchmark {name}: {e}")
                
        df = pd.DataFrame(results).set_index("model")
        return df
