# Telemetry & Reasoning Log

## Purpose
This file tracks the "Intelligence" of the system—the reasoning behind specific psychopolitical heuristics and DSA choices.

## Log Structure
| Timestamp | Component | Decision | Rationale | Harness Mandate |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-09 | Harness | Added TDD Skill | Ensure 100% logic coverage | `harness_prompt.txt` |
| 2026-06-11 | Oracle | Unified models into Consensus Oracle | Stacking Poisson, Bayesian, and Ensemble models via a Ridge meta-learner to reduce individual model bias. | `CORE_PROBLEM_TO_SOLVE.md` |
| 2026-06-11 | Training | Fixed Pipeline Index Bug | Reset indices before feature subtraction to prevent 2x row expansion and ValueError. | `DSA_POLICY.md` |

## Model Benchmarking (2026-06-11)
| Model | RMSE | MAE | Accuracy |
| :--- | :--- | :--- | :--- |
| **Bayesian-Hierarchical** | **1.7673** | **1.2828** | **0.6739** |
| Elo-Only | 1.7833 | 1.2990 | 0.6721 |
| Stacking-Ensemble | 1.8420 | 1.3466 | 0.6678 |
| **CONSENSUS-ORACLE** | **1.9138** | **1.3833** | **0.6497** |
| Double-Poisson | 2.0084 | 1.4421 | 0.5770 |
| Culture-Only | 2.1361 | 1.5643 | 0.5398 |
| Politics-Only | 2.1572 | 1.5856 | 0.5337 |

## Reasoning Log
1. **Consensus Strategy**: The Consensus Oracle uses a Meta-Stacking approach. While the Bayesian-Hierarchical model shows the lowest historical error, the Consensus Oracle is designed to handle the "Psych-Pivot" by integrating absolute goal predictions (Poisson) with contextual/hierarchical differences.
2. **Poisson Stability**: Observation of constant `Poisson_home` predictions in 2026 forecasts suggests either feature starvation in the 2026 dataset or high model regularization. This warrants further investigation into Layer 11 feature density.
3. **Index Alignment**: Critical bug fixed in `oracle_training_pipeline.py`. When differencing home/away features, `X_home - X_away` without index resets caused a Cartesian-like expansion due to duplicate index values from the temporal spine. Standardized on `.reset_index(drop=True)`.

## Usage
Update this file after every "Refactor" phase in TDD or after completing a milestone.
