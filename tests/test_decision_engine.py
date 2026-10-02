import unittest

from src.decision_engine import DecisionEngine


class FakeFirebase:
    def __init__(self):
        self.alerts = []
        self.events = []
        self.live_updates = []

    def send_alert(self, result):
        self.alerts.append(result)

    def log_event(self, result):
        self.events.append(result)

    def update_live_sensors(self, result):
        self.live_updates.append(result)


class FakeArduino:
    def __init__(self):
        self.relay_activations = 0
        self.relay_deactivations = 0

    def activate_relay(self):
        self.relay_activations += 1

    def deactivate_relay(self):
        self.relay_deactivations += 1


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        self.firebase = FakeFirebase()
        self.arduino = FakeArduino()
        self.engine = DecisionEngine(
            firebase_handler=self.firebase,
            arduino_reader=self.arduino,
        )

    def evaluate(self, **overrides):
        values = {
            "gas_label": "air",
            "gas_risk": 0.10,
            "temp_anomaly": False,
            "temp_error": 0.0,
            "fire_prob": 0.0,
            "flame_signal": 0,
        }
        values.update(overrides)
        return self.engine.evaluate(**values)

    def test_safe_classification(self):
        result = self.evaluate()
        self.assertEqual(result["level"], "SAFE")
        self.assertLess(result["score"], 0.35)
        self.assertEqual(self.arduino.relay_activations, 0)

    def test_warning_classification_and_alert(self):
        result = self.evaluate(gas_label="methane", gas_risk=0.80)
        self.assertEqual(result["level"], "WARNING")
        self.assertGreaterEqual(result["score"], 0.35)
        self.assertLess(result["score"], 0.65)
        self.assertEqual(len(self.firebase.alerts), 1)
        self.assertEqual(len(self.firebase.events), 1)
        self.assertEqual(self.arduino.relay_activations, 0)

    def test_critical_classification_activates_relay(self):
        # Exercise weighted-fusion CRITICAL behavior without using the
        # separate CNN emergency override (fire_prob >= 0.80).
        result = self.evaluate(
            gas_label="methane",
            gas_risk=0.90,
            temp_anomaly=True,
            temp_error=0.90,
            fire_prob=0.70,
        )
        self.assertEqual(result["level"], "CRITICAL")
        self.assertEqual(result["response_level"], "CRITICAL")
        self.assertGreaterEqual(result["score"], 0.65)
        self.assertEqual(self.arduino.relay_activations, 1)
        self.assertEqual(len(self.firebase.alerts), 1)
        self.assertEqual(len(self.firebase.events), 1)

    def test_flame_sensor_is_immediate_critical_override(self):
        result = self.evaluate(
            gas_risk=0.0,
            fire_prob=0.0,
            flame_signal=1,
        )
        self.assertEqual(result["level"], "CRITICAL")
        self.assertEqual(result["method"], "flame_sensor")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(self.arduino.relay_activations, 1)

    def test_high_cnn_probability_is_immediate_critical_override(self):
        result = self.evaluate(fire_prob=0.80)
        self.assertEqual(result["level"], "CRITICAL")
        self.assertEqual(result["method"], "cnn_override")
        self.assertEqual(self.arduino.relay_activations, 1)

    def test_current_people_are_preserved_in_result(self):
        people = [{"person_id": "ahad", "name": "Ahad", "designation": "Student"}]
        result = self.evaluate(current_people=people)
        self.assertEqual(result["current_people"], people)

    def test_relay_reset_deactivates_relay(self):
        self.evaluate(fire_prob=0.90)
        self.engine.reset_relay()
        self.assertEqual(self.arduino.relay_deactivations, 1)


if __name__ == "__main__":
    unittest.main()
