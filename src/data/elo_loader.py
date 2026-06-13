import pandas as pd
import logging
import os
from src.data.base_loader import BaseLoader
from src.data.entity_resolver import get_iso3_code

logger = logging.getLogger(__name__)

class EloLoader(BaseLoader):
    """
    Loads Elo ratings and saves to CSV in data/processed/.
    """
    def __init__(self):
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def extract(self) -> pd.DataFrame:
        path = os.path.join("data", "raw", "elo.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
            
        df.columns = [c.lower() for c in df.columns]
        df["year"] = 2026
        df["country_code"] = df["team"].apply(get_iso3_code)
        
        return df.dropna(subset=["country_code"])
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "elo_ratings.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved Elo ratings to {path}")
