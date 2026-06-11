import requests
import pandas as pd
import logging
import os
import zipfile
import io
from typing import Dict

from src.data.base_loader import BaseLoader
from src.data.entity_resolver import get_iso3_code

logger = logging.getLogger(__name__)

class ConflictLoader(BaseLoader):
    """
    Loads conflict data (Layer 7) from UCDP (CSV mirror).
    Saves cleaned data to data/processed/.
    """
    
    UCDP_URL = "https://ucdp.uu.se/downloads/ucdpprio/ucdp-prio-acd-251-csv.zip"
    
    def __init__(self):
        self.raw_dir = os.path.join("data", "raw")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def _fetch_ucdp(self) -> pd.DataFrame:
        csv_filename = "UcdpPrioConflict_v25_1.csv"
        local_csv_path = os.path.join(self.raw_dir, csv_filename)
        
        if os.path.exists(local_csv_path):
            logger.info(f"Loading UCDP data from local cache: {local_csv_path}")
            return pd.read_csv(local_csv_path)
            
        try:
            logger.info("Fetching UCDP data from remote.")
            response = requests.get(self.UCDP_URL, timeout=30)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logger.error("No CSV file found in the UCDP zip.")
                    return pd.DataFrame()
                
                extracted_file = csv_files[0]
                z.extract(extracted_file, self.raw_dir)
                logger.info(f"Extracted {extracted_file} to {self.raw_dir}")
                
                return pd.read_csv(os.path.join(self.raw_dir, extracted_file))
        except Exception as e:
            logger.error(f"UCDP fetch failed: {e}")
            return pd.DataFrame()

    def extract(self) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting conflict data...")
        return {"ucdp": self._fetch_ucdp()}
        
    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        results = []
        
        if "ucdp" in raw_data and not raw_data["ucdp"].empty:
            ucdp = raw_data["ucdp"].copy()
            for _, row in ucdp.iterrows():
                locations = str(row["location"]).split(",")
                for loc in locations:
                    results.append({
                        "country": loc.strip(),
                        "year": int(row["year"]),
                        "conflict_type": "UCDP Armed Conflict",
                        "intensity": float(row["intensity_level"])
                    })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df["country_code"] = df["country"].apply(get_iso3_code)
            df = df.dropna(subset=["country_code"])
            df = df.groupby(["country_code", "year"]).agg({
                "intensity": "max",
                "conflict_type": "first"
            }).reset_index()
            
        return df
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "conflict_data.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved conflict data to {path}")
