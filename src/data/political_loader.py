import wbgapi as wb
import pandas as pd
import logging
import os
import json
from typing import Dict, Any

from src.data.base_loader import BaseLoader

logger = logging.getLogger(__name__)

class PoliticalLoader(BaseLoader):
    """
    Loads political and economic data (Layers 5 & 6) from the World Bank API.
    Saves cleaned data to data/processed/.
    """

    INDICATORS = {
        "political_stability": "GOV_WGI_PV.EST",
        "gdp_per_capita": "NY.GDP.PCAP.CD"
    }

    def __init__(self):
        self.raw_dir = os.path.join("data", "raw")
        self.processed_dir = os.path.join("data", "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def extract(self) -> Dict[str, pd.DataFrame]:
        logger.info("Extracting World Bank data...")
        years = range(2015, 2023)
        cache_file = os.path.join(self.raw_dir, "wb_data.json")

        if os.path.exists(cache_file):
            logger.info("Loading WB data from cache.")
            with open(cache_file, 'r') as f:
                data = json.load(f)
            # Reconstruct DataFrames from cached dicts
            return {k: pd.DataFrame(v) for k, v in data.items()}

        try:
            pol_df = wb.data.DataFrame(self.INDICATORS["political_stability"], time=years, db=3)
            gdp_df = wb.data.DataFrame(self.INDICATORS["gdp_per_capita"], time=years, db=2)

            data_to_cache = {
                "political_stability": pol_df.to_dict(),
                "gdp_per_capita": gdp_df.to_dict()
            }
            with open(cache_file, 'w') as f:
                json.dump(data_to_cache, f)

            return {
                "political_stability": pol_df,
                "gdp_per_capita": gdp_df
            }
        except Exception as e:
            logger.error(f"World Bank extraction failed: {e}")
            return {}

    def transform(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not raw_data:
            return pd.DataFrame()
            
        def melt_wb(df: pd.DataFrame, val_name: str) -> pd.DataFrame:
            df = df.reset_index()
            id_col = df.columns[0]
            melted = df.melt(id_vars=[id_col], var_name="year_raw", value_name=val_name)
            melted["year"] = melted["year_raw"].str.extract(r"(\d{4})").astype(float)
            melted = melted.dropna(subset=["year"])
            melted["year"] = melted["year"].astype(int)
            melted = melted.rename(columns={id_col: "country_code"})
            return melted[["country_code", "year", val_name]]

        pol_melted = melt_wb(raw_data["political_stability"], "political_stability")
        gdp_melted = melt_wb(raw_data["gdp_per_capita"], "gdp_per_capita")
        
        panel_df = pd.merge(pol_melted, gdp_melted, on=["country_code", "year"], how="outer")
        return panel_df
        
    def save_processed(self, df: pd.DataFrame) -> None:
        """
        Saves transformed data to CSV in data/processed/.
        """
        path = os.path.join(self.processed_dir, "political_economic.csv")
        df.to_csv(path, index=False)
        logger.info(f"Saved political/economic data to {path}")
