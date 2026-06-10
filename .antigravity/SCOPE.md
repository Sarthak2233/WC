# Scope Control Policy

## Context Management
- ** Reads:** read more related files at once. Use `grep_search` to target specific lines.
- **Vertical Isolation:** When working on a specific Data Layer (e.g., Layer 5: Political), ignore files related to unrelated layers (e.g., Layer 2: FBref) unless a join is explicitly required.
- **Context Pruning:** At the end of every sub-phase, explicitly "forget" transient variables or local assumptions that are not codified in `.antigravity`.

## Data Handling
- **Staging Isolation:** Always write raw data to `data/raw/` and validated data to `data/processed/`. Never modify raw data in place.
- **Master Table Schema:** Every write to the 7 Master Tables must be validated against the schema defined in `DATASET_CREATION-DERICTIVE.md`.

## Execution Limits
- **Max Files:** Do not modify more than 2 files in a single turn.
- **Max Turns:** If a task takes more than 5 turns without reaching "Green" in TDD, invoke the `/diagnostic-loop`.
