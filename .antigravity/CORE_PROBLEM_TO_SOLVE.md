# Core Problem to Solve: Unified Oracle Engine & Contest Arena

## The Challenge: Predictive Supremacy & Operational Integrity
Building the "World Cup 2026 Oracle & Arena" is a dual challenge of building a **high-performance prediction engine** and a **high-integrity tournament state machine**.

---

## 1. The Predictive Challenge (The Oracle)
The engine must prove that **Psychopolitical Modeling** consistently outperforms standard football analytics (Elo, FIFA rankings, market value).
- **Beating the Baselines:** Success is measured by accuracy against 5-10 synthetic baseline predictors (Elo-only, FIFA-rank-only, Market-value-only, etc.).
- **Signal vs. Noise in 11 Layers:** Efficiently converging 11 layers of volatile news, economic, and cultural data into the **7 Master Tables**.
- **Causal Patterns:** Grounding synthetic features (Adversity Index, Psyche Score) in the 10 core heuristics of the **Causal Pattern Library**.

## 2. The Operational Challenge (The Arena)
The arena is a **Temporal State Machine** that manages the tournament's hard state transitions with absolute rigor.
- **Temporal Integrity:** Every prediction and feature must be strictly timestamped. No "look-ahead bias"—the engine must "time-travel" back to the eve of every match.
- **Hard Locks & Irreversible Transitions:** Once a stage deadline passes or a match result is ingested, the state is locked. There is no rollback.
- **Scoreline Ambiguity:** Resolving the 90m vs. 120m vs. Pens scoring weights (the "Psych-Pivot").
- **Bracket Integrity:** Autonomously validating knockout trees as teams are eliminated.

## 3. Event-Driven Infrastructure
The system treats every prediction, result, and calculation as an **immutable event** in an append-only ledger.
- **Atomic Scoring Pipeline:** Match result ingested $\rightarrow$ fan out to all participants (Oracle + Baselines + Humans) $\rightarrow$ update standings.
- **Model Recalibration:** After each stage, the Oracle autonomously ingests actual results, recomputes the tree, and generates new predictions.

## Technical Mandates
- **Idempotency:** All ETL and feature engineering must be idempotent.
- **DSA Optimization:** Mandatory $O(N)$ joining efficiency across 11 layers and 22 tournaments.
- **Verification:** Every algorithmic and psychopolitical decision must be logged in `TELEMETRY.md`.

**In essence: You are building a system that quantifies the "Human Spirit" to predict scores, while managing the most rigorous tournament arena ever engineered.**
