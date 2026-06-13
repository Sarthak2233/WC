from pathlib import Path
from src.features.feature_converger import FeatureConverger


def test_feature_variance_report():
    repo_root = Path(__file__).resolve().parents[1]
    processed_matches = repo_root / 'data' / 'processed' / 'matches.csv'
    if not processed_matches.exists():
        import pytest
        pytest.skip(f"Processed matches.csv not found at {processed_matches}; run data pipeline first")

    conv = FeatureConverger(processed_dir=str(repo_root / 'data' / 'processed'), master_dir=str(repo_root / 'data' / 'master'), models_dir=str(repo_root / 'models'))
    conv.run()

    report = repo_root / 'data' / 'master' / 'feature_variance_report.csv'
    assert report.exists(), f"Feature variance report not created at {report}"

    import csv
    with report.open() as f:
        reader = csv.DictReader(f)
        allowed_exceptions = ['happiness_score', 'is_host', 'legacy_weight']
        for r in reader:
            # Ensure retained features have acceptable zero_rate unless whitelisted
            kept = r.get('kept', 'True').lower() in ('true', '1', 'yes')
            feat = r.get('feature')
            if kept:
                if feat in allowed_exceptions or feat.startswith('elo_'):
                    continue
                zr = float(r.get('zero_rate', 0))
                assert zr <= 0.95, f"Retained feature {r['feature']} has high zero_rate {zr}"
