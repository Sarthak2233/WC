import pandas as pd
import requests
import logging
import os
from io import StringIO
from typing import Dict

from src.data.base_loader import BaseLoader

logger = logging.getLogger(__name__)

class FifaLoader(BaseLoader):
    """
    Loads raw FIFA player data (Layer 2.5) into a CSV file.
    """
    
    FIFA23_URL = "https://raw.githubusercontent.com/miraehab/FIFA-23-ML-Project/main/players_fifa23.csv"
    
    def __init__(self):
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
        logger.info("Extracting FIFA attribute data...")
        return {"fifa23": self._fetch_csv("players_fifa23.csv")}
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if raw_data["fifa23"].empty:
            return pd.DataFrame()
            
        df = raw_data["fifa23"].copy()
        
        # Mapping verified CSV headers to PlayerRaw fields
        rename_map = {
            "FullName": "full_name",
            "Nationality": "nationality",
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
            "PhysicalityTotal": "physicality_total",
            "BestPosition": "best_position",
            "Height": "height_cm",
            "Weight": "weight_kg",
            "Crossing": "crossing",
            "Finishing": "finishing",
            "ShortPassing": "short_passing",
            "Dribbling": "dribbling",
            "Stamina": "stamina",
            "Strength": "strength",
            "Vision": "vision",
            "Penalties": "penalties",
            "Composure": "composure"
        }
        
        df = df.rename(columns=rename_map)
        
        # Keep only mapped columns
        cols_to_keep = [v for v in rename_map.values() if v in df.columns]
        return df[cols_to_keep]
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves player data to a CSV in data/processed/.
        """
        if df.empty:
            return
        
        path = os.path.join(self.processed_dir, "players_fifa.csv")
        
        # Simple upsert logic: read existing, merge, write
        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            # Merge on full_name and nationality
            df = pd.concat([existing_df, df]).drop_duplicates(subset=["full_name", "nationality"], keep="last")
            
        df.to_csv(path, index=False)
        logger.info(f"Successfully saved/updated Layer 2.5: FIFA Attribute CSV.")
