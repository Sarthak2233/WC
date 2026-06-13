import pandas as pd
import requests
import logging
import os
from io import StringIO
from typing import Dict

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import get_iso3_code

logger = logging.getLogger(__name__)

class CultureLoader(BaseLoader):
    """
    Loads cultural dimensions (Layer 8) and social mood (Layer 9).
    Sources: Hofstede (GitHub/Plotly) and World Happiness Report (GitHub/pplonski).
    Saves cleaned data to data/processed/.
    """
    
    HOFSTEDE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/hofstede-cultural-dimensions.csv"
    HAPPINESS_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/world_happiness_report/WHR_2024.csv"
    
    def __init__(self):
        self.raw_dir = os.path.join("data", "raw")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def _fetch_csv(self, filename: str, url: str, sep: str = ",") -> pd.DataFrame:
        local_path = os.path.join(self.raw_dir, filename)
        if os.path.exists(local_path):
            logger.info(f"Loading {filename} from local cache.")
            return pd.read_csv(local_path, sep=sep)

        try:
            logger.info(f"Fetching {filename} from remote.")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text), sep=sep, na_values=["#NULL!", "NULL", "nan", "N/A"])
            df.to_csv(local_path, index=False, sep=sep)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return pd.DataFrame()

    def extract(self) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting cultural and social mood data...")
        hofstede = self._fetch_csv("hofstede.csv", self.HOFSTEDE_URL, sep=";")
        happiness = self._fetch_csv("happiness.csv", self.HAPPINESS_URL)

        return {
            "hofstede": hofstede,
            "happiness": happiness
        }
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if raw_data["hofstede"].empty and raw_data["happiness"].empty:
            return pd.DataFrame()
            
        # 1. Hofstede
        hof_df = raw_data["hofstede"].copy()
        hof_df["country_code"] = hof_df["country"].apply(get_iso3_code)
        hof_df = hof_df.dropna(subset=["country_code"])
        
        # 2. Happiness
        hap_df = raw_data["happiness"].copy()
        hap_df["country_code"] = hap_df["country"].apply(get_iso3_code)
        hap_df = hap_df.dropna(subset=["country_code"])
        
        # Merge
        merged = pd.merge(
            hof_df[["country_code", "pdi", "idv", "mas", "uai", "ltowvs", "ivr"]],
            hap_df[["country_code", "happiness_score"]],
            on="country_code",
            how="outer"
        )
        
        merged = merged.rename(columns={"ltowvs": "lto"})
        
        return merged
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "culture_happiness.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved culture/happiness data to {path}")
