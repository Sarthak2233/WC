import os
import pandas as pd
import logging
from src.utils.entity_mapper import standardize_country_name

logger = logging.getLogger(__name__)

class FeatureConvergence:
    def __init__(self, processed_dir: str, master_dir: str):
        self.processed_dir = processed_dir
        self.master_dir = master_dir
        os.makedirs(self.master_dir, exist_ok=True)

    def load_processed(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.processed_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def run(self):
        logger.info("Starting Master Feature Convergence...")
        
        # 1. Base Teams (the 48 teams for WC 2026)
        teams_df = self.load_processed("wc_2026_teams.csv")
        if teams_df.empty:
            logger.error("wc_2026_teams.csv not found. Aborting.")
            return

        # 2. Add ELO ratings
        elo_df = self.load_processed("elo.csv")
        if not elo_df.empty:
            teams_df = teams_df.merge(elo_df[['team', 'elo']], on='team', how='left')

        # 3. Add Team Performance Aggregates
        # We need to aggregate the cleaned player performance first
        perf_df = self.load_processed("fifa_world_cup_2026_player_performance.csv")
        if not perf_df.empty:
            logger.info("Aggregating player performance to team level...")
            # Columns to aggregate
            score_cols = [c for c in perf_df.columns if any(m in c for m in ["score", "rating", "resistance", "contribution", "impact", "xg", "xa"])]
            team_perf = perf_df.groupby('team')[score_cols].mean().reset_index()
            # Prefix them
            team_perf.columns = [f"mean_{c}" if c != 'team' else c for c in team_perf.columns]
            teams_df = teams_df.merge(team_perf, on='team', how='left')

        # 4. Add Happiness/Wellbeing
        happy_df = self.load_processed("happiness.csv")
        if not happy_df.empty:
            # Look for country_name column
            country_col = 'country_name' if 'country_name' in happy_df.columns else 'country'
            if country_col in happy_df.columns:
                happy_features = ['life_ladder', 'log_gdp_per_capita', 'social_support', 'freedom_to_make_life_choices']
                cols_to_join = [country_col] + [c for c in happy_features if c in happy_df.columns]
                teams_df = teams_df.merge(happy_df[cols_to_join], left_on='team', right_on=country_col, how='left').drop(columns=[country_col], errors='ignore')

        # 5. Add Cultural Dimensions (Hofstede)
        hof_df = self.load_processed("hofstede.csv")
        if not hof_df.empty:
            # Columns: country, pdi, idv, mas, uai, lto, ivr
            teams_df = teams_df.merge(hof_df, left_on='team', right_on='country', how='left').drop(columns=['country'], errors='ignore')

        # 6. Add Political/Conflict Context
        conflict_df = self.load_processed("UcdpPrioConflict_v25_1.csv")
        if not conflict_df.empty:
            # Simplification: Intensity of conflict for the country (SideA)
            # Find max intensity or presence
            sidea_col = 'sidea' if 'sidea' in conflict_df.columns else 'side_a'
            if sidea_col in conflict_df.columns:
                conflict_summary = conflict_df.groupby(sidea_col)['intensity_level'].max().reset_index()
                teams_df = teams_df.merge(conflict_summary, left_on='team', right_on=sidea_col, how='left').drop(columns=[sidea_col], errors='ignore')

        # 7. Calculate Derived Features (Milestone 4.1, 4.2)
        # Political Pressure Index (PPI) - Synthetic combination of conflict and freedom (from happiness)
        if 'intensity_level' in teams_df.columns and 'freedom_to_make_life_choices' in teams_df.columns:
            # Fill NaNs for PPI calculation
            teams_df['intensity_level'] = teams_df['intensity_level'].fillna(0)
            teams_df['freedom_to_make_life_choices'] = teams_df['freedom_to_make_life_choices'].fillna(teams_df['freedom_to_make_life_choices'].mean())
            teams_df['ppi_synthetic'] = (teams_df['intensity_level'].astype(float) * 0.7) + ((1 - teams_df['freedom_to_make_life_choices'].astype(float)) * 0.3)
            
        # Collective Psyche Score - Synthetic combination of performance metrics and happiness metrics
        if 'mean_clutch_performance_score' in teams_df.columns and 'life_ladder' in teams_df.columns:
             teams_df['collective_psyche'] = (teams_df['mean_clutch_performance_score'].fillna(0) * 0.5) + (teams_df['life_ladder'].fillna(0) * 0.5)

        # 8. Final Clean & Impute
        # Fill remaining NaNs with global means for the matrix
        teams_df = teams_df.fillna(teams_df.mean(numeric_only=True))

        # 9. Save to Master
        output_path = os.path.join(self.master_dir, "oracle_master_features.csv")
        teams_df.to_csv(output_path, index=False)
        logger.info(f"Master feature matrix saved to {output_path}. Shape: {teams_df.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conv = FeatureConvergence("data/processed", "data/master")
    conv.run()
