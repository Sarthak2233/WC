import pandas as pd
import requests
import logging
import os
from io import StringIO
from typing import Dict, Any
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import get_iso3_code
from src.database import Culture

logger = logging.getLogger(__name__)

class CultureLoader(BaseLoader):
    """
    Loads cultural dimensions (Layer 8) and social mood (Layer 9).
    Sources: Hofstede (GitHub/Plotly) and World Happiness Report (GitHub/pplonski).
    """
    
    HOFSTEDE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/hofstede-cultural-dimensions.csv"
    HAPPINESS_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/world_happiness_report/WHR_2024.csv"
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.raw_dir = os.path.join("data", "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

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
        """
        Extracts Hofstede and Happiness datasets.
        """
        logger.info("Extracting cultural and social mood data...")
        # Hofstede uses semicolon delimiter
        hofstede = self._fetch_csv("hofstede.csv", self.HOFSTEDE_URL, sep=";")
        happiness = self._fetch_csv("happiness.csv", self.HAPPINESS_URL)

        return {
            "hofstede": hofstede,
            "happiness": happiness
        }
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms and joins cultural metrics.
        """
        if raw_data["hofstede"].empty and raw_data["happiness"].empty:
            return pd.DataFrame()
            
        # 1. Hofstede
        hof_df = raw_data["hofstede"].copy()
        # Columns: ctr;country;pdi;idv;mas;uai;ltowvs;ivr
        # Standardize country code to ISO3
        hof_df["country_code"] = hof_df["country"].apply(get_iso3_code)
        hof_df = hof_df.dropna(subset=["country_code"])
        
        # 2. Happiness
        hap_df = raw_data["happiness"].copy()
        # Columns: country, region, happiness_score, etc.
        hap_df["country_code"] = hap_df["country"].apply(get_iso3_code)
        # happiness_score is already the name in this CSV mirror
        hap_df = hap_df.dropna(subset=["country_code"])
        
        # Merge
        merged = pd.merge(
            hof_df[["country_code", "pdi", "idv", "mas", "uai", "ltowvs", "ivr"]],
            hap_df[["country_code", "happiness_score"]],
            on="country_code",
            how="outer"
        )
        
        # Rename ltowvs to lto
        merged = merged.rename(columns={"ltowvs": "lto"})
        
        return merged
        
    def load(self, df: pd.DataFrame) -> None:
        """
        Loads transformed data into the Culture table.
        """
        if df.empty:
            return
            
        session: Session = self.session_factory()
        try:
            for _, row in df.iterrows():
                existing = session.query(Culture).filter_by(country_code=row["country_code"]).first()
                
                culture_data = {
                    "country_code": row["country_code"],
                    "pdi": None if pd.isna(row.get("pdi")) else float(row["pdi"]),
                    "idv": None if pd.isna(row.get("idv")) else float(row["idv"]),
                    "mas": None if pd.isna(row.get("mas")) else float(row["mas"]),
                    "uai": None if pd.isna(row.get("uai")) else float(row["uai"]),
                    "lto": None if pd.isna(row.get("lto")) else float(row["lto"]),
                    "ivr": None if pd.isna(row.get("ivr")) else float(row["ivr"]),
                    "happiness_score": None if pd.isna(row.get("happiness_score")) else float(row["happiness_score"])
                }
                
                if not existing:
                    session.add(Culture(**culture_data))
                else:
                    # Update fields
                    for k, v in culture_data.items():
                        if v is not None:
                            setattr(existing, k, v)
                            
            session.commit()
            logger.info("Successfully loaded Layer 8 & 9: Culture and Social Mood.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading culture data: {e}")
            raise
        finally:
            session.close()
