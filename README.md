# Predictive AI Hazard Prevention System
## Final Year Project - Complete Software Package

---

## STEP 1: Install Python 3.10
Download from https://python.org (tick "Add to PATH")

## STEP 2: Install dependencies
```
cd fyp_hazard_system
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

## STEP 3: Run in simulation mode (NO hardware needed)
```
python src/main.py --sim
```
Open browser: http://localhost:5000

## STEP 4: Train the AI models (software only)
```
python notebooks/01_train_gas_model.py     # trains Random Forest
python notebooks/02_train_lstm_model.py    # trains LSTM
# For CNN: download Kaggle fire dataset first (see notebook for instructions)
python notebooks/03_train_cnn_model.py
```

## STEP 5: Configure Firebase
1. Create project at https://console.firebase.google.com
2. Enable Realtime Database
3. Download serviceAccountKey.json -> place in project root
4. Update DATABASE_URL in src/firebase_handler.py
5. Update firebaseConfig in dashboard/templates/register_token.html

## STEP 6: Register admin phone for push notifications
```
python src/main.py --sim   # start system
```
Open on admin phone: http://YOUR_COMPUTER_IP:5000/register
Click "Enable push notifications"

## STEP 7: Run with hardware (when hardware is ready)
```
python src/main.py
```
Or specify port:
```
python src/main.py --port COM4
```

---

## HARDWARE CALIBRATION (do this when hardware arrives)
1. Upload arduino/hazard_sensors.ino to Arduino
2. Open Serial Monitor at 9600 baud
3. In clean air, note the GAS_RATIO values for 5 minutes
4. Average those values - update CLEAN_AIR_VALUE in the .ino file
5. Re-upload sketch
6. Collect labeled gas readings -> retrain gas model

---

## PROJECT STRUCTURE
```
fyp_hazard_system/
  arduino/
    hazard_sensors.ino     <- Upload this to Arduino (ONE file)
  src/
    arduino_reader.py      <- Serial communication + simulation
    gas_model.py           <- Random Forest MQ-2 classifier
    lstm_model.py          <- LSTM temperature anomaly detection
    cnn_model.py           <- CNN MobileNetV2 fire detection
    decision_engine.py     <- Original fusion logic (our contribution)
    firebase_handler.py    <- Cloud DB + FCM notifications (Android+iOS)
    main.py                <- Entry point - runs everything
  dashboard/
    app.py                 <- Flask web server
    templates/
      index.html           <- Live monitoring dashboard
      register_token.html  <- Admin phone FCM registration
    static/js/
      firebase-sw.js       <- Service worker for background notifications
  notebooks/
    01_train_gas_model.py  <- Train Random Forest
    02_train_lstm_model.py <- Train LSTM
    03_train_cnn_model.py  <- Train CNN
  models/                  <- Saved model files (created after training)
  requirements.txt
```

---

## ADJUSTMENT POINTS (marked with ADJUST THIS in code)
- arduino/hazard_sensors.ino: CLEAN_AIR_VALUE (calibrate with your sensor)
- src/firebase_handler.py:    DATABASE_URL
- dashboard/templates/register_token.html: firebaseConfig + VAPID_KEY
- dashboard/static/js/firebase-sw.js:      firebaseConfig

Everything else works as-is.


## Real-time Person Presence and Hazard Correlation

The camera now supports a local person-presence pipeline in addition to fire detection.

### Runtime behavior

1. The camera detects visible faces continuously.
2. Registered faces are recognized locally and annotated above the person's head with:
   - Name
   - Designation
   - Match score
3. Unknown/unregistered faces are labelled **Unknown**.
4. When a person leaves the camera view, the active presence entry expires automatically.
5. When a warning/critical hazard event occurs, the people visible at that moment are attached to the event record.
6. Firebase stores:
   - `/persons` - name/designation metadata
   - `/live_presence` - people currently visible
   - `/live_sensors` - current sensor/prediction state
   - `/events` - hazard events including people present

### Registering people

Run the application and open the dashboard. Use **Register a person** and upload one or more clear face images. The face model is stored locally under `data/person_faces/` and is intentionally ignored by Git.

The dashboard endpoints are:

- `POST /persons/register`
- `GET /persons`
- `POST /reset_relay`

For real hardware, keep the camera pointed at the monitored zone. The system records that a recognized person was visible in the camera's monitored area when a hazard event occurred; it does **not** calculate exact physical distance from the hazard.

### OpenCV requirement

Face recognition uses OpenCV's contrib face module. The requirements file therefore uses `opencv-contrib-python` instead of `opencv-python`.

### Privacy

Biometric face samples and the local recognition model are not committed to GitHub. Firebase receives person metadata and live/event presence information, not the local face-image files.
