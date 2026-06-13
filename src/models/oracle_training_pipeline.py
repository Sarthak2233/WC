import logging
import pandas as pd
import pickle
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from src.features.csv_oracle import CSVFeatureOracle
from src.models.trainer import StageSpecializedTrainer, OracleTrainer, BayesianHierarchicalTrainer, PoissonTrainer, ConsensusOracle, BaselineTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_export_trainer(name, trainer, X, y, source_list, base_path):
    """
    Trains a specific trainer and exports its data with source tracking.
    """
    logger.info(f"Training and tracking: {name}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Trainer {name}: X shape {X.shape}, y shape {y.shape}")

    # Reset indices to ensure boolean masks align
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    if hasattr(y_train, 'reset_index'):
        y_train = y_train.reset_index(drop=True)
    if hasattr(y_test, 'reset_index'):
        y_test = y_test.reset_index(drop=True)

    # Drop rows with NaN targets before training
    if hasattr(y_train, 'isna'):
        mask_train = ~y_train.isna()
        X_train = X_train.loc[mask_train].reset_index(drop=True)
        y_train = y_train.loc[mask_train].reset_index(drop=True)
    if hasattr(y_test, 'isna'):
        mask_test = ~y_test.isna()
        X_test = X_test.loc[mask_test].reset_index(drop=True)
        y_test = y_test.loc[mask_test].reset_index(drop=True)

    # Train
    trainer.train(X_train, y_train)
    
    # Save Metadata
    save_dir = os.path.join(base_path, name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Exported Data
    X_train_export = X_train.copy()
    X_train_export['y'] = y_train
    X_train_export['SOURCE_FILES'] = source_list
    X_train_export.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    
    X_test_export = X_test.copy()
    X_test_export['y'] = y_test
    X_test_export['SOURCE_FILES'] = source_list
    X_test_export.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    
    # Save Model
    with open(os.path.join(save_dir, f"{name.lower()}_model.pkl"), "wb") as f:
        pickle.dump(trainer, f)
        
    logger.info(f"Exported data and model for {name} to {save_dir}/")

def run_training_pipeline():
    logger.info("Initializing multi-trainer training pipeline...")
    
    oracle_engine = CSVFeatureOracle("data/processed")
    X_home, y_home, X_away, y_away, matches = oracle_engine.build_absolute_training_set()
    source_list = ",".join(oracle_engine.loaded_files)
    
    base_path = "models/v3"
    os.makedirs(base_path, exist_ok=True)
    
    # 1. Train Absolute models (Poisson)
    train_and_export_trainer("HomePoisson", PoissonTrainer(), X_home, y_home, source_list, base_path)
    train_and_export_trainer("AwayPoisson", PoissonTrainer(), X_away, y_away, source_list, base_path)
    
    # 2. Train Difference models (Ensemble, Bayesian)
    # Fix: Ensure indices are reset before subtraction to avoid row explosion
    X_diff = X_home.reset_index(drop=True) - X_away.reset_index(drop=True)
    X_diff.columns = [f"diff_{c}" for c in X_diff.columns]
    y_diff = y_home.reset_index(drop=True) - y_away.reset_index(drop=True)
    
    # RESIDUAL LEARNING: Train Elo baseline first
    elo_trainer = BaselineTrainer(feature_cols=["diff_elo_elo"])
    elo_trainer.train(X_diff, y_diff)
    y_elo_preds = elo_trainer.predict(X_diff)
    y_residual = y_diff - y_elo_preds
    logger.info("Residuals calculated using Elo-Only baseline.")
    
    # Add stage_name back for specialized training
    X_diff_with_stage = X_diff.copy()
    X_diff_with_stage['stage_name'] = matches['stage_name'].values
    
    # StageSpecializedTrainer for Ensemble (trained on RESIDUALS)
    ensemble_trainer = StageSpecializedTrainer(
        group_trainer=OracleTrainer(),
        knockout_trainer=OracleTrainer()
    )
    
    train_and_export_trainer("Ensemble", ensemble_trainer, X_diff_with_stage, y_residual, source_list, base_path)
    train_and_export_trainer("Bayesian", BayesianHierarchicalTrainer(), X_diff_with_stage, y_residual, source_list, base_path)
    
    # 3. Train Consensus Oracle (Meta-Stacking)
    # Note: Keep old logic for now to avoid Consensus Oracle breakage
    logger.info("Training Consensus Oracle...")
    indices = np.arange(len(X_home))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_h_tr = X_home.iloc[train_idx].reset_index(drop=True)
    X_a_tr = X_away.iloc[train_idx].reset_index(drop=True)
    y_h_tr = y_home.iloc[train_idx].reset_index(drop=True)
    y_a_tr = y_away.iloc[train_idx].reset_index(drop=True)
    
    X_d_tr = (X_h_tr - X_a_tr)
    X_d_tr.columns = [f"diff_{c}" for c in X_d_tr.columns]
    
    X_m_tr = pd.concat([
        X_d_tr,
        X_h_tr.add_prefix("home_"),
        X_a_tr.add_prefix("away_")
    ], axis=1)
    
    y_m_tr = pd.DataFrame({
        "diff": y_h_tr - y_a_tr,
        "home_goals": y_h_tr,
        "away_goals": y_a_tr
    })
    
    consensus = ConsensusOracle()
    consensus.train(X_m_tr, y_m_tr)
    
    save_dir = os.path.join(base_path, "Consensus")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "consensus_model.pkl"), "wb") as f:
        pickle.dump(consensus, f)
    logger.info(f"Exported Consensus Oracle to {save_dir}/")

    # Save feature names
    with open(os.path.join(base_path, "feature_names.json"), "w") as f:
        json.dump(X_home.columns.tolist(), f)

if __name__ == "__main__":
    run_training_pipeline()
