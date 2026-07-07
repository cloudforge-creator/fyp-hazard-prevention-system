"""
gas_model.py
MQ-2 gas classifier using Random Forest.
MQ-2 detects: LPG, Propane, Hydrogen, Alcohol, Smoke, Methane, CO.

KEY INSIGHT: We use the Rs/Ro ratio from the MQ-2 as our feature.
This is the correct approach for a single MQ sensor - not 128 features.
The model classifies hazard level based on ratio ranges from the MQ-2 datasheet.
"""
import numpy as np
import joblib
import os

MODEL_PATH  = 'models/gas_rf_model.pkl'
SCALER_PATH = 'models/gas_scaler.pkl'

# MQ-2 Rs/Ro ratio ranges from datasheet (approximate)
# Lower ratio = higher gas concentration
GAS_RATIO_THRESHOLDS = {
    'clean_air':  (8.0, float('inf')),
    'lpg':        (1.0, 2.0),
    'methane':    (0.6, 1.5),
    'smoke':      (3.0, 8.0),
    'alcohol':    (0.4, 1.0),
    'hydrogen':   (0.3, 0.8),
    'co':         (1.5, 3.0),
}

# Risk level per gas type for MQ-2
GAS_RISK = {
    'clean_air': 0.0,
    'smoke':     0.4,
    'lpg':       0.7,
    'methane':   0.75,
    'co':        0.8,
    'alcohol':   0.5,
    'hydrogen':  0.85,
}


class GasModel:
    def __init__(self):
        self.model_loaded = False
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model  = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.model_loaded = True
                print("[GasModel] Trained model loaded")
            except Exception as e:
                print(f"[GasModel] Model load failed: {e} - using rule-based")
        else:
            print("[GasModel] No trained model found - using MQ-2 datasheet rules")
            print("[GasModel] Run notebooks/01_train_gas_model.ipynb to train")

    def predict(self, gas_ratio: float, gas_raw: int):
        """
        Predict gas type and risk from MQ-2 ratio reading.
        Uses trained model if available, else datasheet-based rules.

        Args:
            gas_ratio: Rs/Ro ratio from Arduino (float)
            gas_raw:   Raw ADC reading 0-1023 (int)

        Returns:
            gas_label (str), risk_score (float 0-1)
        """
        if gas_ratio is None or gas_ratio <= 0:
            return 'unknown', 0.0

        if self.model_loaded:
            return self._model_predict(gas_ratio, gas_raw)
        else:
            return self._rule_predict(gas_ratio)

    def _model_predict(self, gas_ratio, gas_raw):
        features = np.array([[gas_ratio, gas_raw,
                               gas_ratio ** 2,
                               1.0 / max(gas_ratio, 0.01)]])
        features_scaled = self.scaler.transform(features)
        label = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled).max()
        risk  = GAS_RISK.get(label, 0.5) * proba
        return label, round(float(risk), 3)

    def _rule_predict(self, ratio):
        """Datasheet-based rule fallback when model not trained yet."""
        if ratio > 8.0:
            return 'clean_air', 0.0
        elif ratio > 3.0:
            return 'smoke', round(min(0.9, (8.0 - ratio) / 5.0 * 0.9), 3)
        elif ratio > 1.5:
            return 'co', round(min(0.85, (3.0 - ratio) / 1.5 * 0.85), 3)
        elif ratio > 1.0:
            return 'lpg', round(min(0.8, (1.5 - ratio) / 0.5 * 0.8), 3)
        elif ratio > 0.6:
            return 'methane', round(min(0.85, (1.0 - ratio) / 0.4 * 0.85), 3)
        elif ratio > 0.3:
            return 'hydrogen', round(min(0.9, (0.6 - ratio) / 0.3 * 0.9), 3)
        else:
            return 'high_conc_gas', 0.95


# Singleton instance
_gas_model = None

def get_gas_model():
    global _gas_model
    if _gas_model is None:
        _gas_model = GasModel()
    return _gas_model

def predict_gas(gas_ratio, gas_raw):
    return get_gas_model().predict(gas_ratio, gas_raw)
