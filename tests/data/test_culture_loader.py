import pytest
import os
import pandas as pd
from src.data.culture_loader import CultureLoader

def test_culture_loader_run():
    processed_file = "data/processed/culture_happiness.csv"
    if os.path.exists(processed_file):
        os.remove(processed_file)
        
    loader = CultureLoader()
    # Mocking isn't strictly necessary if we just run it and check results
    loader.run()
    
    assert os.path.exists(processed_file)
    df = pd.read_csv(processed_file)
    assert not df.empty
    assert "country_code" in df.columns
    assert "happiness_score" in df.columns
