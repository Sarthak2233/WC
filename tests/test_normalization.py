import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from src.models.trainer import ModelTrainer

class TestNormalizationPipeline(unittest.TestCase):
    def test_pipeline_output_scale(self):
        # Create dummy skewed data
        data = {
            'feat1': np.random.normal(0, 1, 100),
            'feat2': np.random.exponential(1, 100) * 1000 # High variance
        }
        X = pd.DataFrame(data)
        
        # Pipeline: RobustScaler -> StandardScaler
        pipeline = Pipeline([
            ('robust', RobustScaler()),
            ('std', StandardScaler())
        ])
        
        X_processed = pipeline.fit_transform(X)
        
        # Check that variance is approx 1
        self.assertAlmostEqual(np.var(X_processed[:, 0]), 1.0, delta=0.1)
        self.assertAlmostEqual(np.var(X_processed[:, 1]), 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
