import pandas as pd
import numpy as np
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_inventory(master_path="data/master/oracle_master_features.csv"):
    """
    Analyzes the master feature matrix to expose sparsity and provenance issues.
    """
    logger.info(f"Analyzing feature provenance for {master_path}...")
    
    df = pd.read_csv(master_path)
    
    inventory = []
    
    # Identify feature columns
    cols = [c for c in df.columns if c not in ['canonical_team', 'year', 'country_code_x', 'country_code_y']]
    
    for col in cols:
        # Get non-zero/non-NaN data
        active_data = df[df[col] != 0].dropna(subset=[col])
        
        if active_data.empty:
            first_year = np.nan
            last_year = np.nan
            count = 0
        else:
            first_year = active_data['year'].min()
            last_year = active_data['year'].max()
            count = len(active_data)
            
        missing_rate = df[col].isna().mean()
        zero_rate = (df[col] == 0).mean()
        
        inventory.append({
            "feature_name": col,
            "first_year": first_year,
            "last_year": last_year,
            "active_count": count,
            "missing_rate": missing_rate,
            "zero_rate": zero_rate,
            "unique_vals": df[col].nunique()
        })
        
    inventory_df = pd.DataFrame(inventory)
    inventory_df.to_csv("models/feature_inventory.csv", index=False)
    
    logger.info("--- Feature Provenance Summary ---")
    # Flag suspicious features
    leaky = inventory_df[inventory_df['first_year'] > 2023]
    if not leaky.empty:
        logger.warning(f"LEAKAGE DETECTED: {len(leaky)} features only exist in the future (2026).")
        print(leaky[['feature_name', 'first_year', 'active_count']])
        
    dead = inventory_df[inventory_df['active_count'] == 0]
    if not dead.empty:
        logger.warning(f"DEAD FEATURES: {len(dead)} features are 100% empty/zero.")
        
    logger.info(f"Inventory saved to models/feature_inventory.csv")

if __name__ == "__main__":
    generate_inventory()
