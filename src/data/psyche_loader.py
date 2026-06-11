import pandas as pd
import logging
import os
from typing import List, Dict, Any

from src.data.base_loader import BaseLoader

logger = logging.getLogger(__name__)

class PsycheLoader(BaseLoader):
    """
    Loads player psychological profiles derived from performance metrics.
    Source: data/raw/fifa_world_cup_2026_player_performance.csv
    Saves cleaned data to data/processed/.
    """
    
    def __init__(self):
        self.data_path = os.path.join("data", "raw", "fifa_world_cup_2026_player_performance.csv")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def extract(self) -> pd.DataFrame:
        logger.info(f"Extracting performance data from {self.data_path}...")
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
            
        player_metrics = df.groupby('player_name').agg({
            'consistency_score': 'mean',
            'pressure_resistance': 'mean',
            'clutch_performance_score': 'mean'
        }).reset_index()
        
        player_metrics['adversity_score'] = player_metrics.apply(
            lambda row: max(min(1.0 - (row['pressure_resistance'] + row['clutch_performance_score']) / 200, 1.0), 0.0),
            axis=1
        )
        return player_metrics[['player_name', 'adversity_score']]
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "player_psychology.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved player psychology data to {path}")
