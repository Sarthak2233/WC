# World Cup Oracle

## Overview
The World Cup Oracle is a high-integrity predictive engine designed to forecast outcomes for the FIFA World Cup 2026. It employs a psychopolitical modeling approach, integrating traditional sports metrics (Elo ratings) with tournament context (host nation advantage, championship legacy) to transcend simple talent-based forecasting.

## Key Features
- **Forensic Audit Integrity:** The pipeline has been audited for temporal leakage, ensuring the model never uses future information to train on historical data.
- **Causal Feature Engineering:** Includes context-aware features such as `is_host`, `is_defending_champion`, and `legacy_weight`.
- **Walk-Forward Validation:** Employs a strict "Time Machine" evaluation framework, training models on historical data and predicting out-of-sample past tournaments (2018, 2022) to validate predictive power.
- **Robust Inference:** Supports specific match predictions via `predict_2026.py` with automated team-name standardization and categorical outcome interpretation based on score difference thresholds.

## Project Structure
- `data/`: Contains raw input files (`raw/`) and processed/converged datasets (`processed/`, `master/`).
- `models/`: Stores trained ensemble models, feature manifests, and evaluation results.
- `src/data/`: ETL scripts for processing raw data and synthesizing features (e.g., `elo_generator.py`).
- `src/features/`: Core logic for feature convergence and temporal alignment (`feature_converger.py`).
- `src/models/`: Training pipelines, simulation tools, and evaluation scripts (`time_machine.py`, `ablation_study.py`).
- `src/utils/`: Utility functions for entity resolution and auditing (`entity_mapper.py`, `feature_auditor.py`).

## Quick Start
To generate a prediction for a specific match:
```bash
python3 -m src.models.predict_2026 "Team A" "Team B"
```

To run the full diagnostic suite and audit model validity:
```bash
python3 -m src.models.time_machine
```

## Audit Compliance
This project strictly adheres to a "Falsification-First" mindset. The pipeline is validated against an Elo-only baseline, ensuring that all added psychopolitical signals contribute statistically significant predictive power rather than noise.
