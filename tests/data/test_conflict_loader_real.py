import pytest
import os
import pandas as pd
from src.data.conflict_loader import ConflictLoader

def test_conflict_loader_run():
    # Remove existing processed file if it exists to ensure fresh run
    processed_file = "data/processed/conflict_data.csv"
    if os.path.exists(processed_file):
        os.remove(processed_file)
        
    loader = ConflictLoader()
    
    # Run ETL
    loader.run()
    
    # Check for processed file
    assert os.path.exists(processed_file)
    
    # Read and check content
    df = pd.read_csv(processed_file)
    assert not df.empty
    assert "country_code" in df.columns
    assert "intensity" in df.columns
    
    # Check for historical data (e.g., Ukraine)
    ukr_data = df[df["country_code"] == "UKR"]
    assert not ukr_data.empty

if __name__ == "__main__":
    pytest.main([__file__])
