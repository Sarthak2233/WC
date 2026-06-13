import pytest
from pathlib import Path
import pandas as pd


def test_training_targets_reasonable():
    repo = Path(__file__).resolve().parents[1]
    train_path = repo / 'models' / 'v3' / 'AwayPoisson' / 'train.csv'
    if not train_path.exists():
        pytest.skip('AwayPoisson train.csv not present; run training first')
    df = pd.read_csv(train_path)
    if 'y' not in df.columns:
        pytest.skip('No y column in train.csv')
    p99 = df['y'].quantile(0.99)
    # 99th percentile of goals should be reasonable (<10)
    assert p99 < 10, f'Extreme outliers detected in training target: 99th percentile = {p99}'
