import pandas as pd
import requests
import logging
import os
from io import StringIO
from typing import Dict
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.fifa_database import PlayerRaw, FifaSessionLocal

logger = logging.getLogger(__name__)

class FifaLoader(BaseLoader):
    """
    Loads raw FIFA player data (Layer 2.5) into fifa_oracle.db.
    """
    
    FIFA23_URL = "https://raw.githubusercontent.com/miraehab/FIFA-23-ML-Project/main/players_fifa23.csv"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.raw_dir = os.path.join("data", "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        
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
        
    def load(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
            
        session: Session = self.session_factory()
        try:
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Assuming full_name + nationality is the unique key
                existing = session.query(PlayerRaw).filter_by(
                    full_name=row_dict["full_name"],
                    nationality=row_dict["nationality"]
                ).first()
                
                if not existing:
                    session.add(PlayerRaw(**row_dict))
                else:
                    # Update existing with new values
                    for k, v in row_dict.items():
                        if pd.notna(v):
                            setattr(existing, k, v)
                
            session.commit()
            logger.info("Successfully loaded/updated Layer 2.5: FIFA Attribute Database.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading FIFA data: {e}")
            raise
        finally:
            session.close()
