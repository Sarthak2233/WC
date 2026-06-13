import logging
import argparse
import sys
import os
import pandas as pd
import json
import glob

# 1. Setup unified logging
os.makedirs("logs", exist_ok=True)
log_file = "logs/pipeline.log"

# Configure logging to both file and console
file_handler = logging.FileHandler(log_file, mode='w')
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Set root logger
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger("WC_Oracle_Orchestrator")

def run_data_pipeline():
    logger.info("--- [PHASE 1] Data Ingestion & Transformation ---")
    # Lazy imports to keep CLI fast
    from src.data.football_loader import FootballLoader
    from src.data.political_loader import PoliticalLoader
    from src.data.culture_loader import CultureLoader
    from src.data.conflict_loader import ConflictLoader
    from src.data.psyche_loader import PsycheLoader
    from src.data.elo_loader import EloLoader
    from src.data.performance_loader import PerformanceLoader
    from src.data.squad_loader import SquadLoader
    
    loaders = [
        FootballLoader(), PoliticalLoader(), CultureLoader(), 
        ConflictLoader(), PsycheLoader(), EloLoader(),
        PerformanceLoader(None), SquadLoader()
    ]
    
    for loader in loaders:
        logger.info(f"Executing {loader.__class__.__name__}...")
        try:
            loader.run() # Use the standardized BaseLoader.run()
        except Exception as e:
            logger.error(f"Failed {loader.__class__.__name__}: {e}")
    
    logger.info("Running Squad Processor...")
    from src.features.squad_processor import SquadProcessor
    sp = SquadProcessor()
    sp.process_all_squads()
    
    logger.info("--- [PHASE 2] Feature Convergence ---")
    
    # Isolation: Move 2026 data to avoid training leakage
    import shutil
    os.makedirs("data/processed/2026_only", exist_ok=True)
    files_to_move = glob.glob("data/processed/fifawc26-squadlist-*.csv") + \
                    glob.glob("data/processed/fifa_world_cup_2026_player_performance.csv") + \
                    glob.glob("data/processed/master_squads_2026.csv")
    
    for f in files_to_move:
        dest = os.path.join("data/processed/2026_only", os.path.basename(f))
        shutil.move(f, dest)
        logger.info(f"Isolated {f} for 2026-only use.")
        
    from src.features.feature_converger import FeatureConverger
    converger = FeatureConverger()
    converger.run()
    logger.info("Master matrix built at data/master/oracle_master_features.csv")

def run_training_and_eval():
    logger.info("--- [PHASE 3] Oracle Training & Benchmarking ---")
    from src.models.oracle_training_pipeline import run_training_pipeline
    run_training_pipeline()
    
    logger.info("Running Benchmarking Suite...")
    from src.models.run_benchmarks import run_benchmarking_suite
    run_benchmarking_suite()

def run_predictions():
    logger.info("--- [PHASE 4] Prediction Generation ---")
    from src.models.generate_gs_system_preds import generate_gs_system_predictions
    generate_gs_system_predictions()
    
    from src.models.generate_comparative_predictions import generate_comparative_report
    generate_comparative_report()
    
    logger.info("--- [PHASE 5] Monte Carlo Simulations ---")
    from src.models.run_simulation import run_simulation
    # Match-level simulations
    run_simulation(full_tournament=False)
    
    # Full tournament simulation (1000 iterations for speed in bootstrap, user can run 10k manually)
    logger.info("Running Full Tournament Simulation (1000 runs)...")
    from src.models.full_tournament_sim import run_full_tournament_mc
    run_full_tournament_mc(n_iterations=100000)

def main():
    parser = argparse.ArgumentParser(description="World Cup Oracle: Scratch-to-Finish Orchestrator")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline from scratch.")
    parser.add_argument("--data", action="store_true", help="Run data and feature pipeline.")
    parser.add_argument("--train", action="store_true", help="Run training and benchmarking.")
    parser.add_argument("--sim", action="store_true", help="Run predictions and simulations.")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    try:
        if args.all or args.data:
            run_data_pipeline()
        if args.all or args.train:
            run_training_and_eval()
        if args.all or args.sim:
            run_predictions()
            
        logger.info("Workflow execution finished successfully.")
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
