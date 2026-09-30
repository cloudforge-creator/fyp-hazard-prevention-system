"""decision_engine.py
Multi-model safety decision and response logic.
"""
import time
import threading

WEIGHTS = {'cnn': 0.50, 'rf': 0.35, 'lstm': 0.15}
THRESHOLD_WARNING = 0.35
THRESHOLD_CRITICAL = 0.65
HIGH_RISK_GASES = {'lpg', 'methane', 'hydrogen', 'co', 'high_conc_gas'}
FIRE_OVERRIDE_PROB = 0.80
FLAME_OVERRIDE_SIGNAL = 1


class DecisionEngine:
    """Fuse AI outputs and trigger safety responses."""

    def __init__(self, firebase_handler=None, arduino_reader=None):
        self.firebase = firebase_handler
        self.arduino = arduino_reader
        self._lock = threading.Lock()
        self._last_warning_time = 0
        self._warning_cooldown = 30
        self._cycle_count = 0
        self._current_level = 'SAFE'
        self._critical_active = False
        print("[DecisionEngine] Initialised with weights:", WEIGHTS)

    def evaluate(self, gas_label: str, gas_risk: float,
                 temp_anomaly: bool, temp_error: float,
                 fire_prob: float, flame_signal: int) -> dict:
        self._cycle_count += 1

        if flame_signal == FLAME_OVERRIDE_SIGNAL or fire_prob >= FIRE_OVERRIDE_PROB:
            reason = 'flame_sensor' if flame_signal == 1 else 'cnn_override'
            result = self._build_result('CRITICAL', 1.0, reason,
                                         gas_label, fire_prob,
                                         temp_anomaly, temp_error)
            self._respond(result)
            return result

        if gas_label in HIGH_RISK_GASES:
            gas_score = min(1.0, gas_risk * 1.3)
        else:
            gas_score = gas_risk

        temp_score = min(1.0, temp_error * 2.0) if temp_anomaly else 0.0
        fused = round(min(1.0,
            fire_prob * WEIGHTS['cnn'] +
            gas_score * WEIGHTS['rf'] +
            temp_score * WEIGHTS['lstm']), 4)

        level = ('SAFE' if fused < THRESHOLD_WARNING else
                 'WARNING' if fused < THRESHOLD_CRITICAL else 'CRITICAL')

        result = self._build_result(level, fused, 'weighted_fusion',
                                     gas_label, fire_prob,
                                     temp_anomaly, temp_error)
        self._respond(result)
        return result

    def _build_result(self, level, score, method, gas_label,
                      fire_prob, temp_anomaly, temp_error):
        return {
            'level': level, 'score': score, 'method': method,
            'gas': gas_label, 'fire_prob': round(fire_prob, 3),
            'temp_anomaly': bool(temp_anomaly),
            'temp_error': round(temp_error, 4),
            'timestamp': time.time(), 'cycle': self._cycle_count
        }

    def _respond(self, result):
        level = result['level']

        if level == 'CRITICAL':
            first_critical = False
            with self._lock:
                if not self._critical_active:
                    self._critical_active = True
                    first_critical = True

            print(f"[ENGINE] *** CRITICAL *** score={result['score']:.2f} "
                  f"gas={result['gas']} fire={result['fire_prob']:.0%}")

            # Keep relay asserted; only issue the physical command once.
            if first_critical and self.arduino:
                self.arduino.activate_relay()

            # Avoid sending duplicate push notifications/events every 500 ms.
            if first_critical and self.firebase:
                self.firebase.send_alert(result)
                self.firebase.log_event(result)

        elif level == 'WARNING':
            now = time.time()
            print(f"[ENGINE] WARNING score={result['score']:.2f} gas={result['gas']}")
            with self._lock:
                if now - self._last_warning_time > self._warning_cooldown:
                    self._last_warning_time = now
                    if self.firebase:
                        self.firebase.send_alert(result)
                        self.firebase.log_event(result)

        elif self._cycle_count % 60 == 0 and self.firebase:
            self.firebase.update_live_sensors(result)

    def get_status(self):
        return self._current_level

    def reset_relay(self):
        """Restore power after the hazard has been cleared."""
        if self.arduino:
            self.arduino.deactivate_relay()
        with self._lock:
            self._critical_active = False
            self._current_level = 'SAFE'
        print("[ENGINE] Relay reset - machine power restored")
