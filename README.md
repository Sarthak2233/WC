# World Cup Oracle 2026: Master System Documentation

This document provides a comprehensive overview of the World Cup Oracle, a high-performance psychopolitical prediction system. It explains the architectural "why" behind every file and provides instructions for full system orchestration.

## 1. System Mission & Strategy
The mission is to predict 2026 World Cup outcomes using a "Psychopolitical Oracle" approach. This means we don't just look at FIFA rankings; we model the **causal drivers** of national team performance:
- **Football Fundamentals:** Historical results and Elo ratings.
- **Political/Economic Stability:** Correlating national stability with athletic focus.
- **Cultural Identity:** Hofstede dimensions modeling team cohesion.
- **Psychological Stress:** Real-time conflict data and historical trauma.
- **Performance Excellence:** Player-level performance metrics aggregated to the squad.

## 2. Codebase Justification (The Bigger Picture)

### Data Ingestion Layer (`src/data/`)
- **`base_loader.py`**: Ensures all loaders follow a strict Extract-Transform-Load (ETL) contract.
- **`*_loader.py`**: Specialised scripts for each domain (Political, Culture, etc.). Justification: Isolate API/CSV complexities per domain to prevent "data pollution".
- **`entity_resolver.py`**: Crucial for cross-dataset mapping. It ensures "Cote d'Ivoire" in Dataset A matches "Ivory Coast" in Dataset B.

### Feature Engineering Layer (`src/features/`)
- **`feature_converger.py`**: The **Orchestrator of Truth**. It merges disparate datasets onto a temporal spine. Justification: Prevents data leakage and ensures the model sees the "world state" exactly as it was on a specific match date.
- **`squad_processor.py`**: Transforms player-level data (e.g., FIFA ratings) into squad-level features.

### Modeling & Simulation Layer (`src/models/`)
- **`trainer.py` & `oracle_training_pipeline.py`**: Implements the **Consensus Oracle**. Instead of one model, we train a Bayesian, a Poisson, and an Ensemble model to find a "wisdom of the crowds" probability.
- **`simulator.py`**: The physics engine of the project. It converts team features into goal distributions.
- **`full_tournament_sim.py`**: Encodes the complex FIFA 2026 48-team bracket. Justification: Match-level predictions are insufficient for "who wins the cup"; we need Monte Carlo paths.

### Contest Arena Layer (`src/arena/`)
- **`hard_lock_engine.py`**: Enforces temporal integrity. Justification: A contest platform is useless if users can "predict" after a match starts.
- **`scoring.py`**: Pure logic for point allocation (Exact score vs Result).

### Infrastructure & Orchestration
- **`bootstrap.py`**: The "Red Button". Justification: A complex pipeline needs a single entry point to ensure steps are run in order (Data → Features → Model → Sim).
- **`database.py`**: Append-only ledger for all predictions, ensuring a permanent, unalterable record of system performance.

## 3. How to Run the System from Scratch

The refactored `bootstrap.py` is your primary tool.

### Complete Rebuild
To run everything—from downloading/loading raw data to generating the final 10,000-run simulation:

How to use the system:
   * Total Rebuild: `python3 bootstrap.py --all`
   * Update Data only: `python3 bootstrap.py --data`
   * Train Models only: `python3 bootstrap.py --train`
   * Run Simulations only: `python3 bootstrap.py --sim`


### Component-Wise Execution
If you only want to update specific parts of the pipeline:
- **Update Data & Features:** `python3 bootstrap.py --data`
- **Re-train & Benchmark Models:** `python3 bootstrap.py --train`
- **Run Simulations & Reports:** `python3 bootstrap.py --sim`

## 4. Understanding Output
- **`logs/pipeline.log`**: The comprehensive audit trail for every execution.
- **`data/master/oracle_master_features.csv`**: The converged 17-dimensional feature matrix.
- **`models/mc_win_probabilities.csv`**: The definitive win-probability table for 2026.
- **`models/benchmark_report.csv`**: Integrity proof showing Oracle superiority over baselines.
-- **`/models/prediction_summary.csv`**: Final Group Stage Predictions
-- **`/models/comparative_prediction_report.csv`**: Comparative Prediction Report
-- **`/models/mc_win_probabilities.csv`**: Full Tournament Simulation Results
---
*Created on June 13, 2026. Optimized for high-integrity prediction environments of this 2026 World Cup.*
