import unittest

import numpy as np

from src.cnn_model import CNNModel


class FakeModel:
    def __init__(self, no_fire_probability):
        self.no_fire_probability = no_fire_probability

    def predict(self, batch, verbose=0):
        return np.array([[self.no_fire_probability]], dtype=np.float32)


class CNNProbabilityTests(unittest.TestCase):
    def setUp(self):
        self.instance = CNNModel.__new__(CNNModel)
        self.instance.model = FakeModel(0.9)

    def test_probability_is_inverted_for_fire_class(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertAlmostEqual(self.instance._model_fire_probability(image), 0.1, places=5)

    def test_high_no_fire_probability_means_low_fire_probability(self):
        self.instance.model = FakeModel(0.02)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertAlmostEqual(self.instance._model_fire_probability(image), 0.98, places=5)


if __name__ == "__main__":
    unittest.main()
