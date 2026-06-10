# Coding Conventions & Standards

## Python Environment
- **Version:** Python 3.10+
- **Type Hinting:** Mandatory for all function signatures and complex variables.
- **Mandatory Libraries:** 
    - Football: `statsbombpy`, `worldfootballR` (or scrapers for FBref/Transfermarkt).
    - Political/Economic: `wbgapi` (World Bank), `pycountry`.
    - Culture: `hofstede-insights` (custom data/scraping).
    - ML/NLP: `scikit-learn`, `xgboost`, `transformers` (for bio NLP), `spacy`.
- **Style:** Adhere to PEP 8. Use `ruff` for linting and formatting.
- **Documentation:** Use Google-style docstrings for all modules, classes, and public methods.

## Data & ML Engineering
- **7 Master Tables:** All ETL must converge into: `World Cups`, `Matches`, `Players`, `Countries`, `Culture`, `Conflict`, and `Narratives`.
- **Pandas/NumPy:** Prefer vectorized operations over loops. Use appropriate dtypes (e.g., `category`, `int32`) to minimize memory footprint.
- **Scikit-Learn:** Follow the `fit`/`transform` pipeline pattern.
- **Reproducibility:** Always set a global `RANDOM_SEED` (default: 42) for all stochastic operations.
- **Logging:** Use the standard `logging` library. Do not use `print()` for system logs.

## Project Structure
- `src/data/`: Data ingestion and cleaning scripts.
- `src/features/`: Synthetic feature generation (Adversity, Psyche, etc.).
- `src/models/`: Model training and evaluation logic.
- `src/visualization/`: Dashboard and plotting scripts.
- `tests/`: Pytest suite mirroring the `src/` structure.

## Error Handling
- Use explicit exception handling. Avoid bare `except:`.
- Log full stack traces for unexpected failures during data scraping or ingestion.
