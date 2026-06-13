import pandas as pd
import os
import glob
import logging
import re
from datetime import datetime
from src.utils.entity_mapper import standardize_country_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SquadProcessor:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir
        # Load player performance for market value and other metrics
        self.perf_df = pd.read_csv(os.path.join(self.raw_dir, 'fifa_world_cup_2026_player_performance.csv'))
        
    def _calculate_age(self, dob_str):
        try:
            dob = datetime.strptime(str(dob_str), '%Y-%m-%d')
            return (datetime(2026, 6, 11) - dob).days // 365
        except:
            return 26 
            
    def _get_club_country(self, club_str):
        # Extract country code from (XXX)
        match = re.search(r'\(([^)]+)\)', str(club_str))
        return match.group(1) if match else "Unknown"
            
    def process_all_squads(self):
        squad_files = glob.glob(os.path.join(self.raw_dir, 'fifawc26-squadlist-*.csv'))
        processed_data = []
        
        # Prepare perf lookup (team, name) -> market value
        perf_lookup = self.perf_df.groupby(['team', 'player_name'])['market_value_eur'].mean().reset_index()

        for f in squad_files:
            team_name = os.path.basename(f).replace('fifawc26-squadlist-', '').replace('.csv', '')
            canonical_team = standardize_country_name(team_name)
            
            df = pd.read_csv(f)
            df['age'] = df['DOB'].apply(self._calculate_age)
            df['club_country'] = df['CLUB'].apply(self._get_club_country)
            
            # Map Market Value
            # Note: Need to match names, may be noisy
            df = df.merge(perf_lookup, left_on=['PLAYER NAME', 'NATIONALITY'], right_on=['player_name', 'team'], how='left')
            df['market_value_eur'] = df['market_value_eur'].fillna(df['market_value_eur'].median())
            
            # Metrics
            peak_players = df[(df['age'] >= 24) & (df['age'] <= 29)]
            elite_leagues = df[df['club_country'].isin(['ENG', 'ESP', 'ITA', 'GER', 'FRA'])]
            
            agg_data = {
                'canonical_team': canonical_team,
                'year': 2026,
                'perf_mean_age': df['age'].mean(),
                'perf_peak_density': len(peak_players) / len(df),
                'perf_global_exposure': len(elite_leagues) / len(df),
                'perf_talent_variance': df['market_value_eur'].std() / df['market_value_eur'].mean(),
                'squad_size': len(df)
            }
            processed_data.append(agg_data)
            
        return pd.DataFrame(processed_data)

if __name__ == "__main__":
    processor = SquadProcessor()
    squad_df = processor.process_all_squads()
    squad_df.to_csv("data/processed/master_squads_2026.csv", index=False)
    logger.info("Processed advanced squad data saved to data/processed/master_squads_2026.csv")
