import unittest
import numpy as np
import pandas as pd

# Import the functions from your ml_pipeline module.
# Adjust the import path according to your project structure.
import src
from src import ml_pipeline


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        """
        Create a small dummy DataFrame to use in tests.
        This DataFrame contains two numeric feature columns and the target column 'vomitoxin_ppb'.
        """
        self.df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'vomitoxin_ppb': [10.0, 20.0, 30.0]
        })

    def test_preprocess_data(self):
        """
        Test the preprocess_data function.
          - Checks that the output DataFrame has the same shape as the input.
          - Verifies that the feature columns (excluding the target) are normalized (mean approx. 0 and std approx. 1).
        """
        processed_df, scaler = src.ml_pipeline.preprocess_data(self.df)

        # The processed DataFrame should have the same shape as the original.
        self.assertEqual(processed_df.shape, self.df.shape)

        # Get feature columns only (exclude the target)
        features = processed_df.drop(columns=['vomitoxin_ppb'])

        # Check if the mean of each feature is close to 0 and the standard deviation is close to 1.
        # Note: For small sample sizes, these may not be exactly 0 or 1.
        self.assertTrue(np.allclose(features.mean(), 0, atol=1e-6))
        self.assertTrue(np.allclose(features.std(ddof=0), 1, atol=1e-6))

    def test_build_model(self):
        """
        Test the build_model function by checking if the output layer has a single unit.
        """
        # In this dummy test, we assume there are 2 features.
        model = src.ml_pipeline.build_model(input_dim=2)
        # Check that the model's output dimension is 1 (for regression).
        self.assertEqual(model.output_shape[-1], 1)


if __name__ == '__main__':
    unittest.main()
