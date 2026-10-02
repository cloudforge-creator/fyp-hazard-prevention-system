import unittest

from src.predictive_risk import PredictiveRiskModel


class PredictiveRiskTests(unittest.TestCase):
    def test_insufficient_history_does_not_warn_early(self):
        model = PredictiveRiskModel(history_seconds=30, forecast_seconds=30, sample_interval=0.5)
        result = model.update(0.1, 0.1, 0.0, timestamp=0)
        self.assertEqual(result["trend"], "INSUFFICIENT_HISTORY")
        self.assertFalse(result["early_warning"])
        self.assertEqual(result["predicted_level"], "SAFE")

    def test_rising_signals_produce_rising_trend(self):
        model = PredictiveRiskModel(history_seconds=30, forecast_seconds=30, sample_interval=0.5)
        result = None
        for i in range(6):
            result = model.update(
                gas_score=0.10 + i * 0.04,
                temp_score=0.05 + i * 0.03,
                fire_score=0.02 + i * 0.03,
                timestamp=float(i),
            )
        self.assertEqual(result["trend"], "RISING")
        self.assertGreater(result["predicted_score"], result["current_score"])

    def test_falling_signals_produce_falling_trend(self):
        model = PredictiveRiskModel(history_seconds=30, forecast_seconds=30, sample_interval=0.5)
        result = None
        for i in range(6):
            result = model.update(
                gas_score=0.90 - i * 0.05,
                temp_score=0.80 - i * 0.04,
                fire_score=0.70 - i * 0.04,
                timestamp=float(i),
            )
        self.assertEqual(result["trend"], "FALLING")

    def test_predicted_score_never_hides_current_risk(self):
        model = PredictiveRiskModel(history_seconds=30, forecast_seconds=30, sample_interval=0.5)
        result = None
        for i in range(6):
            result = model.update(
                gas_score=0.90 - i * 0.01,
                temp_score=0.90 - i * 0.01,
                fire_score=0.90 - i * 0.01,
                timestamp=float(i),
            )
        self.assertGreaterEqual(result["predicted_score"], result["current_score"])

    def test_reset_clears_history(self):
        model = PredictiveRiskModel()
        for i in range(6):
            model.update(0.2, 0.1, 0.0, timestamp=float(i))
        self.assertEqual(len(model.history), 6)
        model.reset()
        self.assertEqual(len(model.history), 0)


if __name__ == "__main__":
    unittest.main()
