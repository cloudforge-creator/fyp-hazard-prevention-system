"""
predictive_logger.py

CSV logger for building a real timestamped dataset for future-hazard prediction.
Each row represents one sensor/model evaluation cycle. The logged data can later
be labeled using observed hazard transitions and used to train a supervised
future-risk model.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from typing import Dict


FIELDS = [
    "timestamp",
    "gas_raw",
    "gas_ratio",
    "gas_risk",
    "temperature",
    "humidity",
    "temp_anomaly",
    "temp_error",
    "fire_prob",
    "flame_signal",
    "current_score",
    "current_level",
    "predicted_score",
    "predicted_level",
    "prediction_trend",
]


class PredictiveDataLogger:
    def __init__(self, path: str = "data/predictive_sensor_log.csv"):
        self.path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, sensors: Dict, result: Dict) -> None:
        row = {
            "timestamp": time.time(),
            "gas_raw": sensors.get("gas_raw", 0),
            "gas_ratio": sensors.get("gas_ratio", 0),
            "gas_risk": result.get("gas_risk", 0),
            "temperature": sensors.get("temp", 0),
            "humidity": sensors.get("humidity", 0),
            "temp_anomaly": int(bool(result.get("temp_anomaly", False))),
            "temp_error": result.get("temp_error", 0),
            "fire_prob": result.get("fire_prob", 0),
            "flame_signal": sensors.get("flame", 0),
            "current_score": result.get("score", 0),
            "current_level": result.get("level", "SAFE"),
            "predicted_score": result.get("predicted_score", 0),
            "predicted_level": result.get("predicted_level", "SAFE"),
            "prediction_trend": result.get("prediction_trend", "N/A"),
        }

        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
