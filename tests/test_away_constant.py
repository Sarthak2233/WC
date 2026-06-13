import csv
from pathlib import Path


def test_away_goals_not_constant():
    # Locate prediction_summary.csv relative to repo root
    repo_root = Path(__file__).resolve().parents[1]
    pred_path = repo_root / "models" / "prediction_summary.csv"
    assert pred_path.exists(), f"prediction_summary.csv not found at {pred_path}. Run the pipeline to generate it."

    values = []
    with pred_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Common column names observed: predicted_away_goals
            for key in ("predicted_away_goals", "away_goals", "predicted_away"):
                if key in row and row[key] not in (None, "", "NA"):
                    try:
                        values.append(float(row[key]))
                    except Exception:
                        pass
                    break

    assert values, "No numeric away-goal predictions found in prediction_summary.csv"
    unique = set(values)
    assert len(unique) > 1, f"Away goals are constant across all predictions: {next(iter(unique))}"
