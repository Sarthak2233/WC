import numpy as np
import pandas as pd
from src.models.trainer import ModelTrainer


def test_scaler_roundtrip():
    # Simple numeric dataframe
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]})
    y = pd.Series([0, 1, 0, 1])

    mt = ModelTrainer()
    mt.train(df, y)

    scaled = mt.scaler.transform(df)
    inv = mt.scaler.inverse_transform(scaled)

    assert np.allclose(inv, df.values, atol=1e-6)
