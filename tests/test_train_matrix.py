from src.features.feature_converger import FeatureConverger
from pathlib import Path


def test_train_matrix_has_features(tmp_path):
    # Run converger (it should be idempotent and handle missing files gracefully)
    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / 'models'
    # Ensure clean state
    if models_dir.exists():
        pass

    conv = FeatureConverger(processed_dir=str(repo_root / 'data' / 'processed'), master_dir=str(repo_root / 'data' / 'master'), models_dir=str(models_dir))
    conv.run()

    train_csv = models_dir / 'train.csv'
    assert train_csv.exists(), f"train.csv not created at {train_csv}"

    import csv
    with train_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)

    # Expect at least one feature column besides 'y'
    assert len(header) > 1, f"Training matrix has insufficient columns: {header}"
