"""
decision_engine.py
ORIGINAL CONTRIBUTION - custom multi-model fusion logic.
Fuses outputs from RF gas model, LSTM temperature model,
and CNN fire model into a single risk classification.
"""
import time
import threading

from src.predictive_risk import PredictiveRiskModel


# Model output weights (must sum to 1.0)
WEIGHTS = {
    'cnn':  0.50,   # visual fire - highest weight (most immediate danger)
    'rf':   0.35,   # gas detection
    'lstm': 0.15    # temp anomaly (early warning signal)
}

# Risk thresholds for fused score
THRESHOLD_WARNING  = 0.35
THRESHOLD_CRITICAL = 0.65

# Gases that escalate risk score by 30%
HIGH_RISK_GASES = {'lpg', 'methane', 'hydrogen', 'co', 'high_conc_gas'}

# Immediate override thresholds (skip fusion, go Critical immediately)
FIRE_OVERRIDE_PROB   = 0.80   # CNN fire probability
FLAME_OVERRIDE_SIGNAL = 1     # physical flame sensor (1 = flame detected)


class DecisionEngine:
    """
    Fuses three model outputs every cycle into a risk classification.
    Triggers automated responses on Warning and Critical events.
    """

    def __init__(self, firebase_handler=None, arduino_reader=None):
        self.firebase = firebase_handler
        self.arduino  = arduino_reader
        self._lock              = threading.Lock()
        self._last_warning_time = 0
        self._warning_cooldown  = 30    # seconds between warning alerts
        self._cycle_count       = 0
        self._current_level     = 'SAFE'
        self._predictor = PredictiveRiskModel(history_seconds=30.0, forecast_seconds=30.0, sample_interval=0.5)
        print("[DecisionEngine] Initialised with weights:", WEIGHTS)

    def evaluate(self,
                 gas_label:    str,
                 gas_risk:     float,
                 temp_anomaly: bool,
                 temp_error:   float,
                 fire_prob:    float,
                 flame_signal: int,
                 gas_raw:      float = 0.0,
                 temperature:  float = 25.0) -> dict:
        """
        Main evaluation called every sensor cycle.

        Args:
            gas_label:    detected gas type string
            gas_risk:     gas risk score 0-1 from GasModel
            temp_anomaly: True/False from LSTMModel
            temp_error:   normalised prediction error 0-1
            fire_prob:    fire probability 0-1 from CNNModel
            flame_signal: 1=flame detected, 0=no flame from sensor

        Returns:
            result dict with keys:
                level, score, method, gas, fire_prob,
                temp_anomaly, temp_error, timestamp
        """
        self._cycle_count += 1

        # ---- HARD OVERRIDE: Immediate Critical ----
        # Physical flame sensor triggered OR CNN very confident
        if flame_signal == FLAME_OVERRIDE_SIGNAL or fire_prob >= FIRE_OVERRIDE_PROB:
            reason = 'flame_sensor' if flame_signal == 1 else 'cnn_override'
            return self._build_result(
                level='CRITICAL',
                score=1.0,
                method=reason,
                gas_label=gas_label,
                fire_prob=fire_prob,
                temp_anomaly=temp_anomaly,
                temp_error=temp_error
            )

        # ---- WEIGHTED FUSION ----
        # Scale gas risk: high-risk gases get 30% boost
        if gas_label in HIGH_RISK_GASES:
            gas_score = min(1.0, gas_risk * 1.3)
        else:
            gas_score = gas_risk

        # Temperature anomaly score: only contributes if anomaly detected
        temp_score = min(1.0, temp_error * 2.0) if temp_anomaly else 0.0

        # CNN fire score directly
        cnn_score = fire_prob

        # Weighted fusion
        fused = (cnn_score  * WEIGHTS['cnn'] +
                 gas_score  * WEIGHTS['rf']  +
                 temp_score * WEIGHTS['lstm'])
        fused = round(min(1.0, fused), 4)

        # ---- CURRENT RISK CLASSIFICATION ----
        if fused < THRESHOLD_WARNING:
            level = 'SAFE'
        elif fused < THRESHOLD_CRITICAL:
            level = 'WARNING'
        else:
            level = 'CRITICAL'

        # ---- PREDICTIVE EARLY-WARNING ASSESSMENT ----
        predictive = self._predictor.update(
            gas_score=gas_score,
            temp_score=temp_score,
            fire_score=fire_prob,
        )

        predicted_level = predictive['predicted_level']
        response_level = 'PREDICTIVE_CRITICAL' if (
            predicted_level == 'CRITICAL' and level != 'CRITICAL'
        ) else level

        result = self._build_result(
            level=level,
            score=fused,
            method='weighted_fusion_predictive_trend',
            gas_label=gas_label,
            fire_prob=fire_prob,
            temp_anomaly=temp_anomaly,
            temp_error=temp_error
        )
        result.update({
            'predicted_level': predicted_level,
            'predicted_score': predictive['predicted_score'],
            'prediction_confidence': predictive['confidence'],
            'prediction_trend': predictive['trend'],
            'early_warning': predictive['early_warning'],
            'forecast_horizon_seconds': predictive['horizon_seconds'],
            'response_level': response_level,
            'gas_raw': gas_raw,
            'temperature': temperature,
        })

        # ---- TRIGGER RESPONSES ----
        self._respond(result)
        return result

    def _build_result(self, level, score, method,
                      gas_label, fire_prob, temp_anomaly, temp_error):
        result = {
            'level':       level,
            'score':       score,
            'method':      method,
            'gas':         gas_label,
            'fire_prob':   round(fire_prob, 3),
            'temp_anomaly':temp_anomaly,
            'temp_error':  round(temp_error, 4),
            'timestamp':   time.time(),
            'cycle':       self._cycle_count
        }
        self._current_level = level
        return result

    def _respond(self, result):
        level = result['level']
        response_level = result.get('response_level', level)

        if response_level in ('CRITICAL', 'PREDICTIVE_CRITICAL'):
            if response_level == 'PREDICTIVE_CRITICAL':
                print(f"[ENGINE] *** PREDICTIVE CRITICAL *** current={result['score']:.2f} "
                      f"predicted={result['predicted_score']:.2f} trend={result['prediction_trend']}")
            else:
                print(f"[ENGINE] *** CRITICAL *** score={result['score']:.2f} "
                      f"gas={result['gas']} fire={result['fire_prob']:.0%}")
            # Cut machine power
            if self.arduino:
                self.arduino.activate_relay()
            # Send Firebase alert
            if self.firebase:
                self.firebase.send_alert(result)
                self.firebase.log_event(result)

        elif response_level == 'WARNING':
            print(f"[ENGINE] WARNING score={result['score']:.2f} "
                  f"predicted={result.get('predicted_score', result['score']):.2f} "
                  f"trend={result.get('prediction_trend', 'N/A')} gas={result['gas']}")
            # Rate-limited warning alerts
            now = time.time()
            with self._lock:
                if now - self._last_warning_time > self._warning_cooldown:
                    self._last_warning_time = now
                    if self.firebase:
                        self.firebase.send_alert(result)
                        self.firebase.log_event(result)

        else:
            # Safe - just log occasionally (every 60 cycles = ~30 seconds)
            if self._cycle_count % 60 == 0:
                if self.firebase:
                    self.firebase.update_live_sensors(result)

    def get_status(self):
        return self._current_level

    def reset_relay(self):
        """Call after hazard cleared to restore machine power."""
        if self.arduino:
            self.arduino.deactivate_relay()
        print("[ENGINE] Relay reset - machine power restored")
