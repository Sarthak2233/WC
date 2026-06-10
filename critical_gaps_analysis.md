# Analysis of Critical Project Gaps

After reviewing the current state of the 11-layer database (L1: Football Results, L2: Performance, L5/6: Political/Economic, L7: Conflict, L8/9: Culture/Mood, L10: Psyche, L11: Narratives), here are the top 4 critical gaps:

## 1. Data Integrity & Completeness (Historical)
While I have implemented loaders for all 11 layers, the historical data (1930–2022) is often sparse in specific attributes due to the inherent nature of the datasets.
*   **Gap:** Many historical players in the database still lack granular stats (e.g., precise goal-scoring context, detailed positioning for early tournaments, or accurate club history).
*   **Action:** Conduct a deep-dive data enrichment phase, perhaps by sourcing specific historical archives (like the RSSSF database) to fill gaps in player performance history that the general Fjelstul dataset might lack.

## 2. Temporal & Narrative Resolution (GDELT)
The narrative layer (L11) relies heavily on GDELT's real-time API. I experienced multiple `429 Too Many Requests` errors during the last population attempt.
*   **Gap:** The narrative timeline is fragile and limited to the last 12 months, which is insufficient for a prediction engine looking at historical tournaments and 2026 projections.
*   **Action:** Migrate the narrative data ingestion from the GDELT API to BigQuery (as recommended for large historical exports) or implement a much more persistent caching/retry wrapper that handles multi-year batches incrementally over a longer duration.

## 3. Psychopolitical Feature Engineering
The Oracle currently relies on an `Adversity Score` proxy (keyword-based) and a basic `Clutch Factor` (match metadata).
*   **Gap:** These features are extremely high-level. They don't yet incorporate the *interaction* between cultural, political, and narrative layers (e.g., does a player's childhood adversity impact their performance more in countries experiencing political instability?).
*   **Action:** Build a sophisticated feature interaction pipeline (`src/features/`) that joins these diverse master tables to create higher-order interaction features (e.g., "National Stability Impact on Squad Resilience").

## 4. Simulation & Validation Engine
The Oracle currently exists as a set of loaders and a database, but lacks the infrastructure to run simulations and compare predictions.
*   **Gap:** There is no Arena to run Monte Carlo simulations or benchmark the Oracle against the synthetic baseline predictors (Elo, FIFA Rankings).
*   **Action:** Implement the simulation engine (`Milestone 6`) and a benchmarking suite (`Milestone 5.4`) to start testing the Oracle's LogLoss against historical matches to see if the "psychopolitical" features provide actual predictive value over standard Elo.
