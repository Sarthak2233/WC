import pandas as pd
import numpy as np
import logging
import os
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_elo(rating1, rating2, score1, score2, k=60, home_adv=100):
    """
    Standard Elo rating calculation.
    """
    # Apply home advantage to the first team
    r1 = rating1 + home_adv
    r2 = rating2
    
    # Expected scores
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
    
    # Actual scores
    if score1 > score2:
        s1, s2 = 1, 0
    elif score1 < score2:
        s1, s2 = 0, 1
    else:
        s1, s2 = 0.5, 0.5
        
    # New ratings
    new_r1 = rating1 + k * (s1 - e1)
    new_r2 = rating2 + k * (s2 - e2)
    
    return new_r1, new_r2

def generate_historical_elo():
    """
    Generates historical Elo ratings by replaying all matches from 1930.
    """
    logger.info("Generating historical Elo ratings from match history...")
    
    matches_path = "data/processed/matches.csv"
    if not os.path.exists(matches_path):
        logger.error("matches.csv not found. Run bootstrap/loaders first.")
        return
        
    df = pd.read_csv(matches_path)
    df = df.sort_values(["tournament_year"]) # Ensure chronological order
    
    # Current ratings state
    ratings = {} # team -> current_elo
    history = [] # rows for output: [team, year, elo]
    
    years = sorted(df["tournament_year"].unique())
    
    for year in years:
        year_matches = df[df["tournament_year"] == year]
        
        for _, m in year_matches.iterrows():
            t1 = standardize_country_name(m["home_team"])
            t2 = standardize_country_name(m["away_team"])
            
            # Initialize new teams at 1500
            if t1 not in ratings: ratings[t1] = 1500
            if t2 not in ratings: ratings[t2] = 1500
            
            # Calculate update
            r1, r2 = calculate_elo(ratings[t1], ratings[t2], m["home_team_score"], m["away_team_score"])
            
            # Update state
            ratings[t1] = r1
            ratings[t2] = r2
            
        # Snapshot at the end of the tournament year
        for team, elo in ratings.items():
            history.append({
                "canonical_team": team,
                "year": year,
                "elo": elo
            })
            
    elo_df = pd.DataFrame(history)
    output_path = "data/processed/elo_ratings_historical.csv"
    elo_df.to_csv(output_path, index=False)
    logger.info(f"Historical Elo ratings saved to {output_path}")
    logger.info(f"Covered {len(years)} tournament years and {len(ratings)} teams.")

if __name__ == "__main__":
    generate_historical_elo()
