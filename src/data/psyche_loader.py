import pandas as pd
import logging
import os
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.database import Player, SessionLocal

logger = logging.getLogger(__name__)

class PsycheLoader(BaseLoader):
    """
    Loads player psychological profiles derived from performance metrics.
    Source: data/raw/Sets_More/fifa_world_cup_2026_player_performance.csv
    """
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.data_path = os.path.join("data", "raw", "Sets_More", "fifa_world_cup_2026_player_performance.csv")
        
    def extract(self) -> pd.DataFrame:
        """Extracts performance data from local CSV."""
        logger.info(f"Extracting performance data from {self.data_path}...")
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            return pd.DataFrame()

    def transform(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Derives psychological metrics (adversity/resilience) 
        from performance data (consistency, pressure resistance).
        """
        if df.empty:
            return []
            
        # Example logic: adversity score is inversely proportional to pressure resistance
        # and consistency scores.
        processed = []
        # Group by player to get aggregate scores
        player_metrics = df.groupby('player_name').agg({
            'consistency_score': 'mean',
            'pressure_resistance': 'mean',
            'clutch_performance_score': 'mean'
        }).reset_index()
        
        for _, row in player_metrics.iterrows():
            # Higher pressure resistance/clutch = lower "adversity" need
            score = 1.0 - (row['pressure_resistance'] + row['clutch_performance_score']) / 200
            processed.append({
                "name": row['player_name'],
                "adversity_score": max(min(score, 1.0), 0.0)
            })
        return processed

    def load(self, processed_data: List[Dict[str, Any]]) -> None:
        """Loads adversity scores into wc_oracle (Player)."""
        session = SessionLocal()
        try:
            for item in processed_data:
                players = session.query(Player).filter(Player.name == item['name']).all()
                for p in players:
                    p.adversity_score = float(item["adversity_score"])
            session.commit()
            logger.info("Successfully updated Player database with adversity scores from performance data.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading adversity: {e}")
        finally:
            session.close()
            
    def run(self):
        df = self.extract()
        processed = self.transform(df)
        self.load(processed)
