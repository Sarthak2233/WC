import pandas as pd
import numpy as np

class ChildhoodAdversityCalculator:
    """
    Computes the Childhood Adversity Index for a player based on their country's
    socioeconomic and political conditions during their formative years (ages 5-15).
    """
    
    def calculate(self, birth_year: int, country_gdp_df: pd.DataFrame, 
                  country_conflict_df: pd.DataFrame, global_gdp_distribution: list[float]) -> float:
        """
        Formula: score = ((1 - gdp_rank) * 5) + min(conflict_years, 5)
        Range: 0 to 10
        """
        if pd.isna(birth_year) or birth_year == 0:
            return 5.0 # default to middle if unknown
            
        years = range(birth_year + 5, birth_year + 16)
        
        # GDP factor
        avg_gdp = 10000.0
        if not country_gdp_df.empty:
            relevant_gdp = country_gdp_df[country_gdp_df["year"].isin(years)]
            if not relevant_gdp.empty:
                avg_gdp = relevant_gdp["gdp_per_capita"].mean()
                
        # Calculate percentile rank of this avg_gdp globally
        gdp_rank = 0.5
        if global_gdp_distribution:
            global_arr = np.array(global_gdp_distribution)
            # percentage of global gdps that are strictly less than our avg_gdp
            gdp_rank = (global_arr < avg_gdp).mean()
            
        # Conflict factor
        conflict_years = 0
        if not country_conflict_df.empty:
            relevant_conflicts = country_conflict_df[
                (country_conflict_df["year"].isin(years)) & 
                (country_conflict_df["conflict_intensity"] > 0)
            ]
            conflict_years = len(relevant_conflicts["year"].unique())
            
        # Base score (0 to 10)
        score = ((1.0 - gdp_rank) * 5.0) + min(conflict_years, 5)
        
        # Ensure bounds
        return float(np.clip(score, 0.0, 10.0))
