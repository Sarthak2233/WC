import os
import pandas as pd
import glob
import logging
from src.utils.entity_mapper import standardize_country_name, standardize_player_name

logger = logging.getLogger(__name__)

class Preprocessor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        # Columns likely to contain country/team names
        self.country_cols = ['team', 'nationality', 'country', 'country_name', 'host', 'winner', 'runner_up', 'sidea', 'sideb', 'opponent_team']

    def clean_column_name(self, col: str) -> str:
        return col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")

    def find_all_files(self, extension: str = ".csv") -> list:
        """Recursively finds all files with given extension."""
        return glob.glob(os.path.join(self.raw_dir, "**", f"*{extension}"), recursive=True)

    def process_file(self, path: str):
        """Generalized cleaning for any CSV file."""
        logger.info(f"Preprocessing {path}...")
        try:
            # Handle potential encoding and delimiter issues
            try:
                df = pd.read_csv(path, encoding='utf-8', sep=None, engine='python')
            except Exception:
                df = pd.read_csv(path, encoding='latin1', sep=None, engine='python')

            if df.empty:
                return

            # 1. Standardize columns
            df.columns = [self.clean_column_name(c) for c in df.columns]
            
            # 2. Map countries in every potential column
            for col in df.columns:
                if any(match in col for match in self.country_cols):
                    df[col] = df[col].apply(lambda x: standardize_country_name(str(x)) if pd.notnull(x) else x)
                
            # 3. Standardize player names if exists
            if 'player_name' in df.columns:
                df['player_name'] = df['player_name'].apply(lambda x: standardize_player_name(str(x)) if pd.notnull(x) else x)

            # 4. Extract year from tournament_id if missing
            if 'tournament_id' in df.columns and 'tournament_year' not in df.columns and 'year' not in df.columns:
                try:
                    df['tournament_year'] = df['tournament_id'].apply(lambda x: int(str(x).split("-")[1]))
                except:
                    pass

            # 5. Canonical column renames for compatibility
            rename_map = {
                'home_team_name': 'home_team',
                'away_team_name': 'away_team',
                'home_team_score': 'home_team_score', # already correct but being safe
                'away_team_score': 'away_team_score'
            }
            df = df.rename(columns=rename_map)

            # 6. Fill missing values
            # Numeric: Mean of current column
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Categorical: "Unknown"
            object_cols = df.select_dtypes(include=['object']).columns
            df[object_cols] = df[object_cols].fillna("Unknown")

            # 5. Output to processed folder (preserving directory structure relative to raw_dir)
            rel_path = os.path.relpath(path, self.raw_dir)
            output_path = os.path.join(self.processed_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            df.to_csv(output_path, index=False)
            logger.info(f"Successfully processed {rel_path}")

        except Exception:
            logger.exception(f"Failed to process {path}")

    def run_all(self):
        """Scans all directories and processes everything."""
        all_csvs = self.find_all_files(".csv")
        logger.info(f"Found {len(all_csvs)} CSV files to process.")
        for csv in all_csvs:
            self.process_file(csv)
        
        # Special case: Aggregation (Milestone 3.3 preview)
        self.aggregate_squads()

    def aggregate_squads(self):
        """Creates a single master squad file for 2026 from processed files."""
        processed_squad_paths = glob.glob(os.path.join(self.processed_dir, "**", "fifawc26-squadlist-*.csv"), recursive=True)
        if not processed_squad_paths:
            return
            
        logger.info("Aggregating processed squads into a single master file...")
        squads = []
        for p in processed_squad_paths:
            df = pd.read_csv(p)
            # Add team name from filename if missing
            if 'team' not in df.columns or (df['team'] == "Unknown").all():
                team = os.path.basename(p).replace("fifawc26-squadlist-", "").replace(".csv", "")
                df['team'] = standardize_country_name(team)
            squads.append(df)
            
        master_df = pd.concat(squads, ignore_index=True)
        master_df.to_csv(os.path.join(self.processed_dir, "master_squads_2026.csv"), index=False)
        logger.info("Saved master_squads_2026.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prep = Preprocessor("data/raw", "data/processed")
    prep.run_all()
