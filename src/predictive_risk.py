"""
predictive_risk.py

Early-warning predictive risk assessment based on temporal trends.

This module does NOT claim to predict an exact fire time. It projects the
current multi-sensor risk signals forward over a configurable horizon and
estimates whether the hazard risk is increasing, stable, or decreasing.

The next stage of the FYP can replace this trend forecaster with a supervised
future-hazard model once enough timestamped/labeled hazard-transition data is
collected.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
import math
import time


@dataclass
class RiskSample:
    timestamp: float
    gas_score: float
    temp_score: float
    fire_score: float


class PredictiveRiskModel:
    """Short-horizon early-warning risk forecaster."""

    def __init__(
        self,
        history_seconds: float = 30.0,
        forecast_seconds: float = 30.0,
        sample_interval: float = 0.5,
    ):
        self.history_seconds = history_seconds
        self.forecast_seconds = forecast_seconds
        self.sample_interval = sample_interval
        max_samples = max(10, int(history_seconds / sample_interval) + 2)
        self.history: Deque[RiskSample] = deque(maxlen=max_samples)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _slope(points: list[Tuple[float, float]]) -> float:
        """Least-squares slope; returns units of score/second."""
        if len(points) < 3:
            return 0.0

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)

        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator <= 1e-9:
            return 0.0

        return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator

    def update(
        self,
        gas_score: float,
        temp_score: float,
        fire_score: float,
        timestamp: Optional[float] = None,
    ) -> Dict:
        now = time.time() if timestamp is None else timestamp

        sample = RiskSample(
            timestamp=now,
            gas_score=self._clamp(gas_score),
            temp_score=self._clamp(temp_score),
            fire_score=self._clamp(fire_score),
        )
        self.history.append(sample)

        # Not enough history for a meaningful temporal prediction.
        if len(self.history) < 6:
            current = self._current_score(sample)
            return {
                "current_score": round(current, 4),
                "predicted_score": round(current, 4),
                "predicted_level": self._classify(current),
                "trend": "INSUFFICIENT_HISTORY",
                "confidence": round(len(self.history) / 6.0, 2),
                "early_warning": False,
                "horizon_seconds": self.forecast_seconds,
            }

        samples = list(self.history)
        gas_points = [(s.timestamp, s.gas_score) for s in samples]
        temp_points = [(s.timestamp, s.temp_score) for s in samples]
        fire_points = [(s.timestamp, s.fire_score) for s in samples]

        gas_slope = self._slope(gas_points)
        temp_slope = self._slope(temp_points)
        fire_slope = self._slope(fire_points)

        latest = samples[-1]

        projected_gas = self._clamp(latest.gas_score + gas_slope * self.forecast_seconds)
        projected_temp = self._clamp(latest.temp_score + temp_slope * self.forecast_seconds)
        projected_fire = self._clamp(latest.fire_score + fire_slope * self.forecast_seconds)

        current_score = self._current_score(latest)
        predicted_score = (
            0.35 * projected_gas
            + 0.20 * projected_temp
            + 0.45 * projected_fire
        )

        # Avoid allowing a mathematical projection to hide a currently high risk.
        predicted_score = max(current_score, predicted_score)

        positive_trends = sum(
            slope > 0.002
            for slope in (gas_slope, temp_slope, fire_slope)
        )

        if positive_trends >= 2:
            trend = "RISING"
        elif positive_trends == 0 and all(
            slope < -0.002 for slope in (gas_slope, temp_slope, fire_slope)
        ):
            trend = "FALLING"
        else:
            trend = "STABLE"

        confidence = min(1.0, len(samples) / self.history.maxlen)
        if positive_trends >= 2:
            confidence = min(1.0, confidence + 0.10)

        early_warning = (
            current_score < 0.35
            and predicted_score >= 0.35
            and trend == "RISING"
        )

        return {
            "current_score": round(current_score, 4),
            "predicted_score": round(predicted_score, 4),
            "predicted_level": self._classify(predicted_score),
            "trend": trend,
            "confidence": round(confidence, 2),
            "early_warning": early_warning,
            "horizon_seconds": self.forecast_seconds,
            "gas_slope": round(gas_slope, 6),
            "temp_slope": round(temp_slope, 6),
            "fire_slope": round(fire_slope, 6),
            "projected_gas": round(projected_gas, 4),
            "projected_temp": round(projected_temp, 4),
            "projected_fire": round(projected_fire, 4),
        }

    @staticmethod
    def _current_score(sample: RiskSample) -> float:
        return min(
            1.0,
            0.35 * sample.gas_score
            + 0.20 * sample.temp_score
            + 0.45 * sample.fire_score,
        )

    @staticmethod
    def _classify(score: float) -> str:
        if score < 0.35:
            return "SAFE"
        if score < 0.65:
            return "WARNING"
        return "CRITICAL"

    def reset(self) -> None:
        self.history.clear()
