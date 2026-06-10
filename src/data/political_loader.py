import wbgapi as wb
import pandas as pd
import logging
import os
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from src.data.base_loader import BaseLoader
from src.database import Country

logger = logging.getLogger(__name__)

class PoliticalLoader(BaseLoader):
    """
    Loads political and economic data (Layers 5 & 6) from the World Bank API.
    """

    # GOV_WGI_PV.EST: Political Stability and Absence of Violence/Terrorism (WGI)
    # NY.GDP.PCAP.CD: GDP per capita (current US$)
    INDICATORS = {
        "political_stability": "GOV_WGI_PV.EST",
        "gdp_per_capita": "NY.GDP.PCAP.CD"
    }

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.raw_dir = os.path.join("data", "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Extracts data, caching results locally.
        """
        logger.info("Extracting World Bank data...")
        years = range(2015, 2023)
        cache_file = os.path.join(self.raw_dir, "wb_data.json")

        if os.path.exists(cache_file):
            logger.info("Loading WB data from cache.")
            with open(cache_file, 'r') as f:
                data = json.load(f)
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
        """
        Melts and joins WB data into a unified panel format.
        """
        if not raw_data:
            return pd.DataFrame()
            
        # Helper to melt WB DataFrame
        def melt_wb(df: pd.DataFrame, val_name: str) -> pd.DataFrame:
            # Reset index to get country code (usually 'economy' or 'id')
            df = df.reset_index()
            # The economy/country column name is usually the first column
            id_col = df.columns[0]
            
            # Filter year columns (formatted as YR2000 or 2000)
            melted = df.melt(id_vars=[id_col], var_name="year_raw", value_name=val_name)
            # Extract 4-digit year
            melted["year"] = melted["year_raw"].str.extract(r"(\d{4})").astype(float)
            melted = melted.dropna(subset=["year"])
            melted["year"] = melted["year"].astype(int)
            melted = melted.rename(columns={id_col: "country_code"})
            return melted[["country_code", "year", val_name]]

        pol_melted = melt_wb(raw_data["political_stability"], "political_stability")
        gdp_melted = melt_wb(raw_data["gdp_per_capita"], "gdp_per_capita")
        
        # Merge on country_code and year
        panel_df = pd.merge(pol_melted, gdp_melted, on=["country_code", "year"], how="outer")
        
        return panel_df
        
    def load(self, df: pd.DataFrame) -> None:
        """
        Loads panel data into the Country table.
        """
        if df.empty:
            logger.warning("No political data to load.")
            return

        session: Session = self.session_factory()
        try:
            for _, row in df.iterrows():
                # Skip if both are NaN
                if pd.isna(row["political_stability"]) and pd.isna(row["gdp_per_capita"]):
                    continue
                    
                existing = session.query(Country).filter_by(
                    country_code=row["country_code"],
                    year=row["year"]
                ).first()
                
                if not existing:
                    country_entry = Country(
                        country_code=row["country_code"],
                        year=row["year"],
                        political_stability=None if pd.isna(row["political_stability"]) else float(row["political_stability"]),
                        gdp_per_capita=None if pd.isna(row["gdp_per_capita"]) else float(row["gdp_per_capita"])
                    )
                    session.add(country_entry)
                else:
                    # Update existing if needed
                    if not pd.isna(row["political_stability"]):
                        existing.political_stability = float(row["political_stability"])
                    if not pd.isna(row["gdp_per_capita"]):
                        existing.gdp_per_capita = float(row["gdp_per_capita"])
            
            session.commit()
            logger.info("Successfully loaded Layer 5 & 6: Political & Economic data.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error loading political data: {e}")
            raise
        finally:
            session.close()
