"""
lstm_model.py
LSTM time-series temperature anomaly predictor.
Uses a rolling 50-reading window to predict the next temperature value.
Flags anomaly when prediction error exceeds trained threshold.
"""
import numpy as np
import joblib
import os
import collections
import math

MODEL_PATH     = 'models/lstm_temp_model.h5'
SCALER_PATH    = 'models/lstm_scaler.pkl'
THRESHOLD_PATH = 'models/lstm_threshold.pkl'

WINDOW_SIZE = 50  # number of past readings used per prediction


class LSTMModel:
    def __init__(self):
        self.model_loaded = False
        self.buffer       = collections.deque(maxlen=WINDOW_SIZE)
        self.scaler       = None
        self.threshold    = None
        self.model        = None
        self._fill_counter = 0

        if (os.path.exists(MODEL_PATH) and
                os.path.exists(SCALER_PATH) and
                os.path.exists(THRESHOLD_PATH)):
            try:
                from tensorflow.keras.models import load_model
                self.model     = load_model(MODEL_PATH)
                self.scaler    = joblib.load(SCALER_PATH)
                self.threshold = joblib.load(THRESHOLD_PATH)
                self.model_loaded = True
                print(f"[LSTMModel] Loaded. Threshold={self.threshold:.4f}")
            except Exception as e:
                print(f"[LSTMModel] Load failed: {e} - using rule-based")
        else:
            print("[LSTMModel] No model found - using statistical anomaly detection")
            print("[LSTMModel] Run notebooks/02_train_lstm_model.ipynb to train")
            # Statistical fallback: maintain rolling stats
            self._temp_history = collections.deque(maxlen=200)

    def predict(self, temperature: float):
        """
        Args:
            temperature: current temperature reading (float, Celsius)
        Returns:
            is_anomaly (bool), error_score (float 0-1), predicted_temp (float)
        """
        if temperature is None:
            return False, 0.0, 0.0

        self.buffer.append(temperature)
        self._fill_counter += 1

        if self.model_loaded:
            return self._lstm_predict(temperature)
        else:
            return self._statistical_predict(temperature)

    def _lstm_predict(self, temperature):
        if len(self.buffer) < WINDOW_SIZE:
            remaining = WINDOW_SIZE - len(self.buffer)
            print(f"[LSTMModel] Warming up - need {remaining} more readings")
            return False, 0.0, temperature

        scaled = self.scaler.transform(
            [[v] for v in self.buffer])
        window = np.array(scaled).reshape(1, WINDOW_SIZE, 1)

        pred_scaled  = self.model.predict(window, verbose=0)[0][0]
        pred_actual  = float(self.scaler.inverse_transform([[pred_scaled]])[0][0])

        actual_scaled = float(self.scaler.transform([[temperature]])[0][0])
        error         = abs(pred_scaled - actual_scaled)

        # Normalise error to 0-1 range
        error_score   = min(1.0, error / (self.threshold * 3))
        is_anomaly    = error > self.threshold

        return bool(is_anomaly), round(error_score, 4), round(pred_actual, 2)

    def _statistical_predict(self, temperature):
        """
        Fallback when model not trained.
        Uses mean + 3*std deviation rule on rolling window.
        """
        self._temp_history.append(temperature)
        if len(self._temp_history) < 20:
            return False, 0.0, temperature

        hist  = list(self._temp_history)
        mean  = sum(hist) / len(hist)
        std   = math.sqrt(sum((x - mean)**2 for x in hist) / len(hist))
        std   = max(std, 0.1)  # avoid division by zero

        z_score    = abs(temperature - mean) / std
        is_anomaly = z_score > 3.0
        error_score = min(1.0, z_score / 5.0)

        return bool(is_anomaly), round(error_score, 4), round(mean, 2)


# Singleton
_lstm_model = None

def get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        _lstm_model = LSTMModel()
    return _lstm_model

def predict_anomaly(temperature):
    return get_lstm_model().predict(temperature)
