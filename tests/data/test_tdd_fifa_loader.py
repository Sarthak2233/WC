import pytest
import pandas as pd
import os
from src.data.fifa_loader import FifaLoader

def test_fifa_loader_upsert():
    loader = FifaLoader()
    
    # Ensure starting clean
    output_path = os.path.join("data", "processed", "players_fifa.csv")
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Mock data: 1 new player, 1 existing player to update
    data = pd.DataFrame([
        {"full_name": "Test Player", "nationality": "Argentina", "overall": 80},
        {"full_name": "Lionel Messi", "nationality": "Argentina", "overall": 95}
    ])
    
    # Run load (populate first)
    loader.save_processed(data)
    
    # Verify population
    df = pd.read_csv(output_path)
    player = df[df["full_name"] == "Test Player"]
    assert not player.empty
    assert player.iloc[0]["overall"] == 80
    
    # Run load (update)
    update_data = pd.DataFrame([
        {"full_name": "Lionel Messi", "nationality": "Argentina", "overall": 99}
    ])
    loader.save_processed(update_data)
    
    # Verify update
    df = pd.read_csv(output_path)
    messi = df[df["full_name"] == "Lionel Messi"]
    assert not messi.empty
    assert messi.iloc[0]["overall"] == 99
