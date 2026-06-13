import pandas as pd
import requests
import logging
import os
from io import StringIO
from typing import Dict, Any, List

from src.data.base_loader import BaseLoader


logger = logging.getLogger(__name__)

class SquadLoader(BaseLoader):
    """
    Loads detailed squad and player metadata (Layer 3).
    Sources: FIFA 23 Kaggle mirror.
    """
    
    # Mirror for FIFA 23 player data
    FIFA23_URL = "https://raw.githubusercontent.com/miraehab/FIFA-23-ML-Project/main/players_fifa23.csv"
    
    def __init__(self, session_factory=None):
        self.raw_dir = os.path.join("data", "raw")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def _fetch_csv(self, filename: str) -> pd.DataFrame:
        local_path = os.path.join(self.raw_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"Loading {filename} from local cache.")
            return pd.read_csv(local_path)
            
        try:
            logger.info(f"Fetching {filename} from remote.")
            response = requests.get(self.FIFA23_URL, timeout=20)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(local_path, index=False)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {filename}: {e}")
            return pd.DataFrame()

    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Extracts player metadata.
        """
        logger.info("Extracting squad info...")
        fifa23 = self._fetch_csv("players_fifa23.csv")
        return {"fifa23": fifa23}
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms FIFA data into player attributes.
        """
        if raw_data["fifa23"].empty:
            return pd.DataFrame()
            
        df = raw_data["fifa23"].copy()
        
        # Mapping based on verified CSV headers
        rename_map = {
            "FullName": "name",
            "Nationality": "country",
            "Club": "club",
            "Overall": "overall",
            "Potential": "potential",
            "Age": "age",
            "ValueEUR": "value_eur",
            "WageEUR": "wage_eur",
            "PaceTotal": "pace_total",
            "ShootingTotal": "shooting_total",
            "PassingTotal": "passing_total",
            "DribblingTotal": "dribbling_total",
            "DefendingTotal": "defending_total",
            "PhysicalityTotal": "physicality_total"
        }
        
        df = df.rename(columns=rename_map)
        
        # Validate required columns
        required = ["name", "country"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}. Columns available: {df.columns.tolist()}")
            return pd.DataFrame()
            
        return df
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        if df.empty:
            return
        path = os.path.join(self.processed_dir, "players_fifa23.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved squad info to {path}")
