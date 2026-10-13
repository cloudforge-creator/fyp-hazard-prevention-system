import unittest

from src.gas_model import GasModel


class GasModelRuleTests(unittest.TestCase):
    def setUp(self):
        self.model = GasModel.__new__(GasModel)
        self.model.model_loaded = False

    def test_invalid_ratio_is_unknown(self):
        self.assertEqual(self.model.predict(None, 0), ("unknown", 0.0))
        self.assertEqual(self.model.predict(0, 0), ("unknown", 0.0))

    def test_clean_air_has_zero_risk(self):
        label, risk = self.model.predict(9.0, 0)
        self.assertEqual(label, "clean_air")
        self.assertEqual(risk, 0.0)

    def test_low_ratio_is_high_concentration_risk(self):
        label, risk = self.model.predict(0.2, 900)
        self.assertEqual(label, "high_conc_gas")
        self.assertEqual(risk, 0.95)

    def test_risk_is_bounded(self):
        for ratio in (0.2, 0.5, 0.8, 1.2, 2.0, 5.0, 10.0):
            _, risk = self.model.predict(ratio, 500)
            self.assertGreaterEqual(risk, 0.0)
            self.assertLessEqual(risk, 1.0)


if __name__ == "__main__":
    unittest.main()
