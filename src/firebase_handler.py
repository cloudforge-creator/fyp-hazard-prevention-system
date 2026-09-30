"""
firebase_handler.py
Handles Firebase Realtime Database and FCM operations.
"""
import threading
from datetime import datetime

DATABASE_URL = 'https://YOUR-PROJECT-rtdb.firebaseio.com'
SERVICE_ACCT_KEY = 'serviceAccountKey.json'


class FirebaseHandler:
    def __init__(self, simulation=False):
        self.simulation = simulation
        self.initialised = False
        self._token_cache = None
        self._lock = threading.Lock()

        if simulation:
            print("[Firebase] SIMULATION mode - no cloud writes")
            return

        if DATABASE_URL == 'https://YOUR-PROJECT-rtdb.firebaseio.com':
            print("[Firebase] DATABASE_URL not configured - using simulation")
            self.simulation = True
            return

        try:
            import firebase_admin
            from firebase_admin import credentials, db
            if not firebase_admin._apps:
                cred = credentials.Certificate(SERVICE_ACCT_KEY)
                firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
            self.db = db
            self.events_ref = db.reference('/events')
            self.sensors_ref = db.reference('/live_sensors')
            self.token_ref = db.reference('/admin_token')
            self.initialised = True
            print("[Firebase] Connected to Firebase Realtime Database")
        except FileNotFoundError:
            print("[Firebase] serviceAccountKey.json not found - simulation mode")
            self.simulation = True
        except Exception as e:
            print(f"[Firebase] Init error: {e} - simulation mode")
            self.simulation = True

    def update_live_sensors(self, data: dict):
        """Overwrite /live_sensors with the complete current sensor state."""
        if self.simulation:
            return
        try:
            payload = {
                'gas_type': data.get('gas', 'unknown'),
                'gas_risk': data.get('gas_risk', data.get('score', 0)),
                'temperature': data.get('temperature', data.get('temp', 0)),
                'humidity': data.get('humidity', 0),
                'fire_prob': data.get('fire_prob', 0),
                'risk_level': data.get('risk_level', data.get('level', 'SAFE')),
                'risk_score': data.get('risk_score', data.get('score', 0)),
                'method': data.get('method', ''),
                'temp_anomaly': data.get('temp_anomaly', False),
                'temp_error': data.get('temp_error', 0),
                'cycle': data.get('cycle', 0),
                'updated_at': datetime.now().isoformat()
            }
            self.sensors_ref.set(payload)
        except Exception as e:
            print(f"[Firebase] Sensor update error: {e}")

    def log_event(self, result: dict):
        if self.simulation:
            print(f"[SIM-Firebase] Event logged: {result['level']} "
                  f"gas={result['gas']} score={result['score']}")
            return
        try:
            self.events_ref.push({
                'level': result['level'],
                'gas': result['gas'],
                'score': result['score'],
                'method': result['method'],
                'fire_prob': result['fire_prob'],
                'temp_anomaly': result['temp_anomaly'],
                'temp_error': result['temp_error'],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[Firebase] Log error: {e}")

    def send_alert(self, result: dict):
        if self.simulation:
            print("[SIM-Firebase] PUSH NOTIFICATION:")
            print(f"  Title: {result['level']} HAZARD DETECTED")
            print(f"  Body: Gas={result['gas']} | Risk={result['score']:.0%} | "
                  f"Fire={result['fire_prob']:.0%}")
            return

        token = self._get_device_token()
        if not token:
            print("[Firebase] No device token - notification not sent")
            return

        try:
            from firebase_admin import messaging
            is_critical = result['level'] == 'CRITICAL'
            message = messaging.Message(
                notification=messaging.Notification(
                    title=f"{'*** ' if is_critical else ''}{result['level']} HAZARD DETECTED",
                    body=(f"Gas: {result['gas'].upper()} | "
                          f"Risk: {result['score']:.0%} | "
                          f"Fire: {result['fire_prob']:.0%} | "
                          f"{datetime.now().strftime('%H:%M:%S')}")
                ),
                android=messaging.AndroidConfig(
                    priority='high',
                    notification=messaging.AndroidNotification(
                        sound='default', channel_id='hazard_alerts',
                        color='#FF0000' if is_critical else '#FFA500',
                        priority=messaging.AndroidNotificationPriority.MAX)),
                apns=messaging.APNSConfig(
                    headers={'apns-priority': '10'},
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=f"{result['level']} HAZARD",
                                body=(f"Gas: {result['gas']} | Risk: {result['score']:.0%}")),
                            sound='default', badge=1, content_available=True))),
                token=token)
            response = messaging.send(message)
            print(f"[Firebase] Notification sent: {response}")
        except Exception as e:
            print(f"[Firebase] Notification error: {e}")

    def _get_device_token(self):
        with self._lock:
            if self._token_cache:
                return self._token_cache
            try:
                snap = self.token_ref.get()
                if snap:
                    self._token_cache = snap
                    return snap
            except Exception as e:
                print(f"[Firebase] Token fetch error: {e}")
            return None

    def get_last_events(self, n=20):
        if self.simulation:
            return []
        try:
            snap = (self.events_ref.order_by_key().limit_to_last(n).get())
            return list(snap.values()) if snap else []
        except Exception as e:
            print(f"[Firebase] Events fetch error: {e}")
            return []
