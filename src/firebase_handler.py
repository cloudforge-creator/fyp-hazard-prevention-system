"""
firebase_handler.py
Handles all Firebase operations:
  - Realtime Database writes (live sensors + event log)
  - FCM push notifications (Android + iOS via APNs)

SETUP STEPS (do these before running):
  1. Go to console.firebase.google.com
  2. Create project -> Enable Realtime Database -> Start in test mode
  3. Project Settings -> Service accounts -> Generate new private key
  4. Save the downloaded JSON as 'serviceAccountKey.json' in project root
  5. For iOS notifications: Project Settings -> Cloud Messaging -> 
     Upload your APNs auth key (.p8 file) from Apple Developer account
  6. Copy your database URL from Firebase console (Realtime Database tab)
"""
import time
import threading
from datetime import datetime


# ---- CONFIGURATION - update these two values ----
DATABASE_URL     = 'https://YOUR-PROJECT-rtdb.firebaseio.com'  # CHANGE THIS
SERVICE_ACCT_KEY = 'serviceAccountKey.json'                    # keep this
# ------------------------------------------------


class FirebaseHandler:
    def __init__(self, simulation=False):
        self.simulation    = simulation
        self.initialised   = False
        self._token_cache  = None
        self._lock         = threading.Lock()

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
                firebase_admin.initialize_app(cred, {
                    'databaseURL': DATABASE_URL
                })
            self.db           = db
            self.events_ref   = db.reference('/events')
            self.sensors_ref  = db.reference('/live_sensors')
            self.token_ref    = db.reference('/admin_token')
            self.initialised  = True
            print("[Firebase] Connected to Firebase Realtime Database")
        except FileNotFoundError:
            print("[Firebase] serviceAccountKey.json not found - simulation mode")
            print("[Firebase] Download it from Firebase Console -> Project Settings -> Service Accounts")
            self.simulation = True
        except Exception as e:
            print(f"[Firebase] Init error: {e} - simulation mode")
            self.simulation = True

    # ---- LIVE SENSOR DATA ----

    def update_live_sensors(self, data: dict):
        """Write current sensor readings to /live_sensors (overwrites, not push)."""
        if self.simulation:
            return  # silent in simulation
        try:
            payload = {
                'gas_type':    data.get('gas', 'unknown'),
                'gas_risk':    data.get('score', 0),
                'temperature': data.get('temp', 0),
                'fire_prob':   data.get('fire_prob', 0),
                'risk_level':  data.get('level', 'SAFE'),
                'temp_anomaly':data.get('temp_anomaly', False),
                'updated_at':  datetime.now().isoformat()
            }
            self.sensors_ref.set(payload)
        except Exception as e:
            print(f"[Firebase] Sensor update error: {e}")

    # ---- EVENT LOG ----

    def log_event(self, result: dict):
        """Push an event record to /events (appends with unique key)."""
        if self.simulation:
            print(f"[SIM-Firebase] Event logged: {result['level']} "
                  f"gas={result['gas']} score={result['score']}")
            return
        try:
            self.events_ref.push({
                'level':       result['level'],
                'gas':         result['gas'],
                'score':       result['score'],
                'method':      result['method'],
                'fire_prob':   result['fire_prob'],
                'temp_anomaly':result['temp_anomaly'],
                'temp_error':  result['temp_error'],
                'timestamp':   datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[Firebase] Log error: {e}")

    # ---- PUSH NOTIFICATIONS ----

    def send_alert(self, result: dict):
        """
        Send push notification via FCM to registered admin device.
        Works for BOTH Android and iOS (via APNs configured in Firebase Console).
        """
        if self.simulation:
            print(f"[SIM-Firebase] PUSH NOTIFICATION:")
            print(f"  Title: {result['level']} HAZARD DETECTED")
            print(f"  Body:  Gas={result['gas']} | "
                  f"Risk={result['score']:.0%} | "
                  f"Fire={result['fire_prob']:.0%}")
            return

        token = self._get_device_token()
        if not token:
            print("[Firebase] No device token - notification not sent")
            print("[Firebase] Open the admin web app on the admin phone to register")
            return

        try:
            from firebase_admin import messaging

            # Determine icon/sound based on severity
            is_critical = result['level'] == 'CRITICAL'

            message = messaging.Message(
                notification=messaging.Notification(
                    title=f"{'*** ' if is_critical else ''}"
                          f"{result['level']} HAZARD DETECTED",
                    body=(f"Gas: {result['gas'].upper()} | "
                          f"Risk: {result['score']:.0%} | "
                          f"Fire: {result['fire_prob']:.0%} | "
                          f"{datetime.now().strftime('%H:%M:%S')}")
                ),

                # Android specific config
                android=messaging.AndroidConfig(
                    priority='high',           # wake device immediately
                    notification=messaging.AndroidNotification(
                        sound='default',
                        channel_id='hazard_alerts',
                        color='#FF0000' if is_critical else '#FFA500',
                        priority=messaging.AndroidNotificationPriority.MAX
                    )
                ),

                # iOS (APNs) specific config
                apns=messaging.APNSConfig(
                    headers={'apns-priority': '10'},  # immediate delivery
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=f"{result['level']} HAZARD",
                                body=(f"Gas: {result['gas']} | "
                                      f"Risk: {result['score']:.0%}")
                            ),
                            sound='default',
                            badge=1,
                            content_available=True
                        )
                    )
                ),

                token=token
            )

            response = messaging.send(message)
            print(f"[Firebase] Notification sent: {response}")

        except Exception as e:
            print(f"[Firebase] Notification error: {e}")

    # ---- DEVICE TOKEN ----

    def _get_device_token(self):
        """Fetch admin device FCM token from Firebase database."""
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
        """Fetch last N events from Firebase for dashboard display."""
        if self.simulation:
            return []
        try:
            snap = (self.events_ref
                    .order_by_key()
                    .limit_to_last(n)
                    .get())
            if not snap:
                return []
            return list(snap.values())
        except Exception as e:
            print(f"[Firebase] Events fetch error: {e}")
            return []
