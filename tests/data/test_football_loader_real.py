import pytest
import os
import pandas as pd
from src.data.football_loader import FootballLoader

def test_football_loader_run():
    # Define processed files
    processed_files = [
        "data/processed/world_cups.csv",
        "data/processed/matches.csv",
        "data/processed/players.csv"
    ]
    
    # Clean up
    for f in processed_files:
        if os.path.exists(f):
            os.remove(f)
            
    loader = FootballLoader()
    loader.run()
    
    # Verify
    for f in processed_files:
        assert os.path.exists(f)
        df = pd.read_csv(f)
        assert not df.empty
