import pandas as pd
from src.features.csv_oracle import CSVFeatureOracle

oracle_engine = CSVFeatureOracle("data/processed")
matrix_2026 = oracle_engine.build_2026_matrix()
teams = matrix_2026["canonical_team"].unique().tolist()

missing_teams = ["Curaçao", "Cabo Verde", "Jordan", "Uzbekistan", "DR Congo", "Argentina", "Germany", "Spain", "Austria", "Portugal", "Saudi Arabia", "Algeria", "Ivory Coast"]

for team in missing_teams:
    if team not in teams:
        print(f"Missing: {team}")
    else:
        print(f"Found: {team}")
