import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_step(command, description):
    logger.info(f"Running: {description}")
    try:
        subprocess.run(command, check=True, shell=True)
        logger.info(f"Finished: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}: {e}")
        sys.exit(1)

def main():
    # 1. Run Data Loaders (Generating CSVs in data/processed)
    # Using python -c to run the loaders programmatically
    loaders = [
        "FootballLoader", "PoliticalLoader", "CultureLoader", 
        "ConflictLoader", "PsycheLoader", "EloLoader"
    ]
    
    for loader in loaders:
        # We need to instantiate and run. Assuming a script-based approach for now.
        # Ideally, we'd have a unified loader runner.
        run_step(f"python3 -c 'from src.data.{loader.lower().replace('loader','_loader')} import {loader}; l={loader}(); df=l.extract(); saved=l.transform(df); l.save_processed(saved)'", f"Running {loader}")
        
    # 2. Train Model
    run_step("python3 -m src.models.oracle_training_pipeline", "Training Oracle Model")
    
    # 3. Predict
    run_step("python3 -m src.models.predict_2026", "Generating 2026 Predictions")

if __name__ == "__main__":
    main()
