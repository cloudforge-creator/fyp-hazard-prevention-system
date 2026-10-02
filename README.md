# Predictive AI Hazard Prevention System

## Final Year Project — AI-Based Real-Time Hazard Detection, Prediction & Prevention

> **Project type:** Final Year Project (FYP)  
> **Primary software stack:** Python, TensorFlow/Keras, Scikit-learn, OpenCV, Flask, Firebase, Docker, GitHub Actions  
> **Embedded layer:** Arduino + MQ-2 + DHT22 + flame sensor + relay  
> **Repository:** `cloudforge-creator/fyp-hazard-prevention-system`

---

## 1. Project Overview

The **Predictive AI Hazard Prevention System** is a multi-layer safety system designed to detect hazardous conditions, combine multiple AI/sensor signals into a single risk decision, provide early-warning prediction, monitor people present in the camera view, and trigger prevention/notification actions.

The software pipeline is:

```
Sensors + Camera
       |
       v
Data Acquisition
       |
       +------------------+------------------+
       |                  |                  |
       v                  v                  v
   Gas RF Model       LSTM Temperature    CNN Fire Model
       |                  |                  |
       +------------------+------------------+
                          |
                          v
                 Decision Engine
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Current Risk             Predictive Risk
       SAFE/WARNING/            Trend / Early
       CRITICAL                 Warning / Forecast
              |                       |
              +-----------+-----------+
                          |
                          v
                 Response Layer
             +------------+------------+
             |            |            |
             v            v            v
          Relay       Firebase     Dashboard
          Control     Events/FCM    + Camera
                          |
                          v
                   Person Presence
                   + Event Correlation
```

The system is intentionally designed so that **simulation mode can run the complete software pipeline without physical hardware**.

---

# 2. Main Objectives

The project software is built to:

1. Read environmental sensor data.
2. Detect/classify gas-related risk.
3. Detect temperature anomalies.
4. Detect visible fire using computer vision.
5. Detect and recognize people locally from the camera stream.
6. Fuse multiple model outputs into one risk score.
7. Classify the current state as **SAFE**, **WARNING**, or **CRITICAL**.
8. Forecast short-term risk using temporal trends.
9. Generate early-warning information before risk becomes critical.
10. Activate the relay response when a critical condition is detected.
11. Store live/event information in Firebase when configured.
12. Send Firebase Cloud Messaging alerts when configured.
13. Display live data, events, risk state, and camera output through a Flask dashboard.
14. Provide software validation through automated tests, Docker build validation, and end-to-end simulation.

---

# 3. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming | Python 3.10 | Main application and AI pipeline |
| ML/DL | TensorFlow 2.13 / Keras | CNN and LSTM models |
| ML | Scikit-learn | Random Forest gas classification |
| Data | NumPy, Pandas, SciPy | Numerical processing and datasets |
| Computer Vision | OpenCV + opencv-contrib | Fire detection, camera processing, face recognition |
| Web | Flask | Real-time monitoring dashboard and REST endpoints |
| Database/Cloud | Firebase Realtime Database | Live data and event storage |
| Notifications | Firebase Cloud Messaging | Push notifications |
| Hardware interface | PySerial | Python ↔ Arduino serial communication |
| Embedded | Arduino C/C++ | Sensor acquisition and relay control |
| Packaging | Docker | Reproducible software environment |
| Testing | Pytest | Unit tests |
| CI | GitHub Actions | Automated software validation |
| Version control | Git + GitHub | Source control and collaboration |

---

# 4. Project Development Flow — From Scratch to End

## Phase 1 — Requirements and System Design

The first stage was to define the safety problem and divide the system into software and embedded components.

### Main software modules

- Sensor data acquisition
- Gas classification
- Temperature anomaly detection
- Fire detection
- Person recognition/presence
- Multi-model decision engine
- Predictive risk assessment
- Firebase integration
- Flask dashboard
- Event logging
- Relay response
- Testing and CI

### Main design principle

Instead of relying on one AI model, the system combines independent signals:

```
Gas Risk
   +
Temperature Anomaly
   +
Fire Probability
   +
Flame Signal
   |
   v
Multi-Model Decision Engine
   |
   v
Unified Hazard Decision
```

---

# 5. Phase 2 — Data Acquisition

The embedded side is represented by Arduino.

The Arduino sketch is:

```
arduino/hazard_sensors.ino
```

It is designed for:

- MQ-2 gas sensor
- DHT22 temperature/humidity sensor
- Flame sensor
- Relay

The Arduino sends sensor values through serial communication.

Python receives those values through:

```
pyserial
```

The Python communication layer is:

```
src/arduino_reader.py
```

### Sensor data flow

```
MQ-2 ------------------+
DHT22 -----------------+--> Arduino --> Serial --> Python
Flame Sensor -----------+
Relay <-----------------+
```

The reader also supports **simulation mode**, which allows development and testing without an Arduino.

---

# 6. Phase 3 — Gas AI Model

## Random Forest Gas Classification

File:

```
src/gas_model.py
```

Training script:

```
notebooks/01_train_gas_model.py
```

The gas pipeline uses the MQ-2 sensor's gas ratio as an important feature.

Runtime features include:

- gas ratio
- ratio squared
- inverse ratio
- logarithm of ratio
- approximate ADC value

The trained model and preprocessing artifacts are stored under:

```
models/
```

Important files include:

```
models/gas_rf_model.pkl
models/gas_scaler.pkl
models/gas_labels.pkl
```

The model produces:

```
gas_label
gas_risk
```

Example gas categories represented in the implementation include:

- clean air
- smoke
- LPG
- methane
- CO
- hydrogen
- alcohol

A rule-based fallback is also implemented when trained model files are unavailable.

---

# 7. Phase 4 — Temperature AI Model

## LSTM Temperature Anomaly Detection

File:

```
src/lstm_model.py
```

Training script:

```
notebooks/02_train_lstm_model.py
```

The LSTM model processes temperature history and identifies abnormal temperature behavior.

The runtime uses the trained model together with preprocessing/threshold artifacts:

```
models/lstm_temp_model.h5
models/lstm_scaler.pkl
models/lstm_threshold.pkl
```

The result contributes:

```
temp_anomaly
temp_error
```

During startup, the model needs historical readings before a meaningful temporal assessment is available.

---

# 8. Phase 5 — CNN Fire Detection

## MobileNetV2-Based Fire Detection

File:

```
src/cnn_model.py
```

Training script:

```
notebooks/03_train_cnn_model.py
```

The computer-vision component uses TensorFlow/Keras and MobileNetV2.

Input size:

```
224 x 224
```

The model distinguishes:

- fire
- no fire

The implementation also contains additional visual processing:

- center ROI analysis
- HSV flame-colored pixel detection
- connected-component analysis
- temporal visual confirmation
- camera frame annotation

Relevant saved model artifacts include:

```
models/cnn_fire_model.h5
models/cnn_fire_best.h5
models/cnn_fire_weights.h5
models/cnn_fire_architecture.json
```

The visual signal is treated as a safety input to the decision engine rather than as a claim of perfectly calibrated probability.

---

# 9. Phase 6 — Person Presence and Recognition

File:

```
src/person_recognition.py
```

The camera pipeline also supports local person presence.

Technology:

- OpenCV
- Haar cascade face detection
- OpenCV LBPH face recognition

The system can:

1. Detect visible faces.
2. Recognize registered people locally.
3. Label unknown/unregistered people.
4. Track active presence.
5. Expire presence when a person leaves the view.
6. Attach people visible at the time of a warning/critical event.

Person registration is available through the dashboard.

### API

```
POST /persons/register
GET  /persons
```

Local biometric files are intentionally kept outside the public source-control workflow.

The system records **presence in the monitored camera view**. It does not calculate an exact physical distance from a person to the hazard.

---

# 10. Phase 7 — Multi-Model Decision Engine

File:

```
src/decision_engine.py
```

This is the central decision-making component and the project's custom software contribution.

The engine combines:

- CNN fire probability
- Random Forest gas risk
- LSTM temperature anomaly/error

### Fusion weights

```
CNN fire       = 0.50
RF gas         = 0.35
LSTM temp      = 0.15
```

The weighted score is conceptually:

```
Risk Score =
    (CNN × 0.50)
  + (Gas × 0.35)
  + (Temperature × 0.15)
```

### Risk thresholds

```
0.00 – <0.35  -> SAFE
0.35 – <0.65  -> WARNING
0.65 – 1.00   -> CRITICAL
```

High-risk gas categories receive an additional risk boost in the implemented logic.

### Immediate fire override

The engine can bypass normal weighted fusion and immediately classify the state as critical when:

- the physical flame signal is active, or
- CNN fire probability reaches the configured override threshold.

---

# 11. Phase 8 — Predictive Risk / Early Warning

File:

```
src/predictive_risk.py
```

The project includes a short-horizon temporal risk forecaster.

It keeps a rolling history of:

- gas score
- temperature score
- fire score

The current implementation:

1. Collects recent readings.
2. Calculates temporal slopes using least-squares regression.
3. Projects the signals forward.
4. Calculates predicted risk.
5. Classifies the predicted state.
6. Determines whether the trend is rising, stable, or falling.
7. Generates an early-warning condition when appropriate.

Configured horizon:

```
History: 30 seconds
Forecast: 30 seconds
Sampling interval: approximately 0.5 seconds
```

### Important scientific scope

This component is a **short-horizon predictive risk / early-warning forecaster**. It does not claim to predict an exact future fire time.

A future research version could replace this trend forecaster with a supervised future-hazard model after collecting sufficient timestamped and labeled hazard-transition data.

---

# 12. Phase 9 — Response and Prevention Layer

When the Decision Engine detects a critical condition, the response layer can:

### 1. Activate relay

```
CRITICAL
   |
   v
ArduinoReader.activate_relay()
   |
   v
RELAY:ON
```

In simulation mode, the system prints that machine power would be cut instead of controlling physical hardware.

### 2. Send Firebase alert

When Firebase is configured, the system sends an alert through the Firebase handler.

### 3. Log hazard event

The event can contain:

- timestamp
- risk level
- score
- gas
- fire probability
- temperature information
- predicted level
- trend
- people present

Warning and critical alerts are rate-limited by cooldown logic.

The dashboard also provides:

```
POST /reset_relay
```

for manual relay reset after the hazard has cleared.

---

# 13. Phase 10 — Firebase Cloud Integration

File:

```
src/firebase_handler.py
```

Firebase is used for cloud-connected monitoring and notification.

The implementation supports storing information such as:

```
/persons
/live_presence
/live_sensors
/events
```

### Firebase responsibilities

- Live sensor state
- Live person presence
- Hazard event history
- Person metadata
- Push notification delivery

### Configuration

Firebase credentials are intentionally not committed to the repository.

Typical configuration includes:

- Firebase service account credentials
- Realtime Database URL
- Web Firebase configuration
- FCM/VAPID configuration

For development and software testing, Firebase can be disabled or simulated.

---

# 14. Phase 11 — Flask Web Dashboard

Main dashboard file:

```
dashboard/app.py
```

The dashboard is served by Flask.

Default address:

```
http://localhost:5000
```

The Flask server exposes the live monitoring interface and API endpoints.

### Main endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Main monitoring dashboard |
| `/data` | GET | Current live sensor/model data |
| `/events` | GET | Recent hazard events |
| `/video` | GET | Camera MJPEG stream |
| `/health` | GET | Application health |
| `/persons` | GET | Registered/current people data |
| `/persons/register` | POST | Register a person |
| `/reset_relay` | POST | Reset relay |

The dashboard is designed to show live system state while the sensor/AI loop runs in the background.

---

# 15. Phase 12 — Main Application Integration

Entry point:

```
src/main.py
```

This file connects all major components.

### Runtime sequence

Every sensor cycle approximately follows:

```
1. Read sensor/camera data
        |
2. Gas Random Forest prediction
        |
3. LSTM temperature analysis
        |
4. CNN fire analysis
        |
5. Person presence update
        |
6. Decision Engine fusion
        |
7. Predictive risk update
        |
8. Dashboard live-data update
        |
9. Event/notification handling
        |
10. Firebase live updates when configured
```

The application starts the sensor loop in the background and runs the Flask dashboard in the main process.

---

# 16. Phase 13 — Simulation Mode

Simulation mode is one of the most important software-development features because it allows the full application to be exercised without physical hardware.

Run:

```bash
python src/main.py --sim
```

Simulation mode provides:

- simulated sensor readings
- simulated Arduino interface
- simulated relay response
- simulated Firebase behavior when configured for simulation
- real AI/model loading
- real Decision Engine execution
- real predictive-risk processing
- real Flask dashboard
- real event processing

This makes it possible to validate the software pipeline independently from physical hardware availability.

---

# 17. Phase 14 — Local Software Validation

The project was validated through a complete simulation run.

The demonstrated run completed:

```
150 cycles
```

The run showed:

- Application startup
- Simulation-mode Arduino reader
- Model initialization
- Gas model loading
- LSTM model loading
- CNN model loading
- Camera pipeline initialization
- Flask dashboard startup
- Successful dashboard API requests
- Predictive warning states
- Critical hazard state
- Simulated relay activation
- Simulated Firebase notification
- Hazard event logging
- Recovery to SAFE state
- Completion of the full cycle sequence

Example critical event observed during simulation:

```
CRITICAL
score = 1.00
gas = methane
fire = 100%
People = 5
```

The simulation then returned to SAFE after the simulated hazard condition cleared.

This demonstrates the software integration path from input → AI → decision → prediction → response → dashboard/event handling.

---

# 18. Phase 15 — Automated Software Testing

The repository contains Pytest tests:

```
tests/
  test_decision_engine.py
  test_predictive_risk.py
```

Run locally:

```bash
python -m pytest -q
```

The tests focus on the core software decision and predictive-risk logic.

---

# 19. Phase 16 — GitHub Actions CI

Workflow:

```
.github/workflows/software-validation.yml
```

The CI pipeline performs software validation automatically.

Current workflow includes:

1. Checkout repository.
2. Set up Python.
3. Install `requirements.txt`.
4. Run Pytest.
5. Build the Docker image.

Docker validation command used by CI:

```bash
docker build -t fyp-hazard-prevention:ci .
```

This provides a repeatable software-quality gate whenever the repository is updated.

---

# 20. Phase 17 — Dockerization

File:

```
Dockerfile
```

The application is containerized using:

```
python:3.10-slim
```

The image installs the Python dependencies and copies:

```
src/
dashboard/
models/
```

The container exposes port:

```
5000
```

The default container command runs the software in simulation mode without requiring Firebase:

```
python src/main.py --sim --no-firebase --web-port 5000
```

### Build

```bash
docker build -t fyp-hazard-prevention .
```

### Run

```bash
docker run --rm -p 5000:5000 fyp-hazard-prevention
```

Then open:

```
http://localhost:5000
```

---

# 21. Complete Project Structure

Current repository structure:

```
fyp-hazard-prevention-system/
│
├── .github/
│   └── workflows/
│       └── software-validation.yml
│
├── arduino/
│   └── hazard_sensors.ino
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py
│   ├── static/
│   │   └── js/
│   │       └── firebase-sw.js
│   └── templates/
│       ├── index.html
│       └── register_token.html
│
├── data/
│   ├── gas_dataset/
│   └── temperature_dataset/
│
├── docs/
│   └── SOFTWARE_VALIDATION.md
│
├── models/
│   ├── cnn_fire_architecture.json
│   ├── cnn_fire_best.h5
│   ├── cnn_fire_model.h5
│   ├── cnn_fire_model_backup.h5
│   ├── cnn_fire_weights.h5
│   ├── gas_labels.pkl
│   ├── gas_rf_model.pkl
│   ├── gas_scaler.pkl
│   ├── lstm_scaler.pkl
│   ├── lstm_temp_model.h5
│   └── lstm_threshold.pkl
│
├── notebooks/
│   ├── 01_train_gas_model.py
│   ├── 02_train_lstm_model.py
│   └── 03_train_cnn_model.py
│
├── src/
│   ├── arduino_reader.py
│   ├── cnn_model.py
│   ├── decision_engine.py
│   ├── firebase_handler.py
│   ├── gas_model.py
│   ├── lstm_model.py
│   ├── main.py
│   ├── person_recognition.py
│   ├── predictive_logger.py
│   └── predictive_risk.py
│
├── tests/
│   ├── test_decision_engine.py
│   └── test_predictive_risk.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

# 22. How to Install From Scratch

## Step 1 — Clone the repository

```bash
git clone https://github.com/cloudforge-creator/fyp-hazard-prevention-system.git
cd fyp-hazard-prevention-system
```

## Step 2 — Create virtual environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 23. Run the Complete Software Without Hardware

The recommended first run is simulation mode:

```bash
python src/main.py --sim
```

Expected startup includes:

```
Mode: SIMULATION
[ArduinoReader] SIMULATION mode - no hardware needed
[Main] All components initialised
[Dashboard] Starting at http://0.0.0.0:5000
```

Open:

```
http://localhost:5000
```

Check:

```
http://localhost:5000/health
http://localhost:5000/data
http://localhost:5000/events
```

---

# 24. Train the AI Models

Training scripts are provided under `notebooks/`.

## Gas Random Forest

```bash
python notebooks/01_train_gas_model.py
```

Output artifacts are stored under `models/`.

## LSTM Temperature Model

```bash
python notebooks/02_train_lstm_model.py
```

Output artifacts are stored under `models/`.

## CNN Fire Model

```bash
python notebooks/03_train_cnn_model.py
```

The CNN training script requires the appropriate fire/no-fire image dataset described by the training script.

> Existing trained model artifacts are already present in the repository, so normal runtime testing can use the saved models without retraining.

---

# 25. Firebase Setup

Firebase is optional for local software simulation.

When cloud functionality is required:

1. Create a Firebase project.
2. Enable Realtime Database.
3. Configure the Firebase service account securely.
4. Configure the database URL.
5. Configure the web Firebase settings.
6. Configure FCM/VAPID settings for browser push notifications.
7. Start the application with Firebase enabled.

**Do not commit service-account credentials, private keys, or other secrets to GitHub.**

For software-only testing, Firebase can be disabled/simulated.

---

# 26. Person Registration

Start the application and open the dashboard.

The registration API accepts:

```
POST /persons/register
```

Required information:

- person ID
- name
- designation
- one or more face images

The local recognition data is maintained under the local `data/` area and is intentionally not treated as public repository content.

After registration, the system can recognize the person locally and associate current presence with hazard events.

---

# 27. Run With Arduino Hardware

When the physical embedded system is ready:

```bash
python src/main.py
```

Or specify the serial port:

```bash
python src/main.py --port COM4
```

The Arduino communication uses:

```
9600 baud
```

The Python application attempts to discover a compatible serial port automatically when one is not supplied.

---

# 28. Arduino Hardware Layer

File:

```
arduino/hazard_sensors.ino
```

Current intended pin mapping:

| Component | Pin |
|---|---|
| MQ-2 gas | A0 |
| DHT22 | D2 |
| Flame sensor | D4 |
| Relay | D7 |

The Arduino sketch sends sensor readings approximately every 500 ms and accepts relay commands:

```
RELAY:ON
RELAY:OFF
```

---

# 29. Hardware Calibration — Separate From Software Completion

The physical MQ-2 sensor requires calibration.

Typical workflow:

1. Upload the Arduino sketch.
2. Open Serial Monitor at 9600 baud.
3. Allow the sensor to stabilize in clean air.
4. Observe gas-ratio readings.
5. Determine the appropriate clean-air reference.
6. Update the calibration constant in the Arduino sketch.
7. Re-upload the sketch.
8. Collect representative labeled sensor data if retraining is required.

Physical calibration, physical relay testing, real sensor accuracy testing, and other laboratory validation are **hardware activities** and are separate from the software implementation/validation documented in this repository.

---

# 30. Command-Line Options

The main application supports options such as:

```bash
python src/main.py --sim
python src/main.py --port COM4
python src/main.py --no-firebase
python src/main.py --web-port 5000
```

Examples:

### Simulation without Firebase

```bash
python src/main.py --sim --no-firebase
```

### Simulation on another web port

```bash
python src/main.py --sim --web-port 8000
```

---

# 31. Software Validation Checklist

| Component | Software status |
|---|---|
| Python application startup | Implemented |
| Simulation mode | Implemented |
| Arduino serial abstraction | Implemented |
| Simulated sensor input | Implemented |
| Gas Random Forest inference | Implemented |
| LSTM temperature analysis | Implemented |
| CNN fire inference | Implemented |
| Camera processing | Implemented |
| Person presence | Implemented |
| Person recognition | Implemented |
| Multi-model decision engine | Implemented |
| Risk thresholds | Implemented |
| Predictive risk trend | Implemented |
| Early-warning logic | Implemented |
| Relay command interface | Implemented |
| Firebase integration | Implemented |
| Firebase event logging | Implemented |
| Push-notification path | Implemented |
| Flask dashboard | Implemented |
| Live data API | Implemented |
| Event API | Implemented |
| Camera streaming API | Implemented |
| Person registration API | Implemented |
| Relay reset API | Implemented |
| Unit tests | Implemented |
| GitHub Actions CI | Implemented |
| Docker build validation | Implemented |
| End-to-end simulation | Completed |
| Software documentation | Completed |

---

# 32. What Was Actually Verified

The software has been exercised through an end-to-end simulation run reaching 150 cycles.

Observed behavior included:

```
Startup
  ↓
Simulation sensor data
  ↓
Gas model loaded
  ↓
LSTM model loaded
  ↓
CNN model loaded
  ↓
Decision Engine
  ↓
Predictive trend processing
  ↓
Flask dashboard
  ↓
API requests
  ↓
WARNING states
  ↓
CRITICAL state
  ↓
Simulated relay activation
  ↓
Simulated Firebase notification/event
  ↓
SAFE recovery
  ↓
150 cycles completed
```

Automated software validation is also configured through GitHub Actions for unit tests and Docker build validation.

---

# 33. Software Completion Scope

For the purpose of **software-only project tracking**, the repository contains the implemented application, AI inference pipeline, predictive risk component, person-presence pipeline, dashboard/API layer, Firebase integration, testing infrastructure, Dockerization, CI workflow, and validation documentation.

### Included in software scope

- Application source code
- AI/ML inference
- AI model artifacts
- Data-processing logic
- Decision Engine
- Predictive risk
- Person recognition software
- Dashboard
- REST endpoints
- Firebase integration
- Notification logic
- Relay-control software interface
- Simulation mode
- Unit tests
- Docker
- GitHub Actions
- Documentation

### Not used to reduce software completion

The following are physical/hardware validation tasks:

- Physical Arduino validation
- MQ-2 laboratory calibration
- DHT22 physical accuracy validation
- Physical flame-sensor validation
- Physical relay/machine-power testing
- Physical camera placement validation

These are separate from the software implementation status.

---

# 34. Important Technical Limitations

### Predictive model

The current predictive component is a temporal trend forecaster. It should be described as **predictive risk / early warning**, not as an exact predictor of when a fire will occur.

### Gas classification

MQ-2 is a broad gas sensor. Gas labels and risk values should be interpreted according to the project's calibration/data assumptions and should not be presented as laboratory-grade gas concentration measurements.

### Person recognition

The camera pipeline identifies presence/registered faces within its visual field. It does not calculate exact physical distance to a hazard.

### Model accuracy

Successful runtime inference and integration testing do not by themselves establish scientific model accuracy. Formal accuracy/precision/recall/F1/AUC evaluation requires appropriate labeled test datasets and experimental methodology.

---

# 35. Troubleshooting

## Port 5000 already in use

Run the application on another port:

```bash
python src/main.py --sim --web-port 8000
```

Then open:

```
http://localhost:8000
```

## No Arduino detected

Use simulation mode:

```bash
python src/main.py --sim
```

## Firebase configuration problem

Run without Firebase:

```bash
python src/main.py --sim --no-firebase
```

## Models not found

Run the relevant training script or verify that the required files exist under:

```
models/
```

## Test failures

Run:

```bash
python -m pytest -q
```

and inspect the failing test before changing the application logic.

---

# 36. Development and Git Workflow

The repository was developed using feature/fix branches and pull requests before final integration into `main`.

The final repository contains:

- feature integration
- software validation changes
- automated CI
- final documentation

Recommended workflow for future changes:

```
Create branch
     ↓
Implement change
     ↓
Run local tests
     ↓
Run simulation
     ↓
Commit
     ↓
Push branch
     ↓
Open Pull Request
     ↓
GitHub Actions validation
     ↓
Review
     ↓
Merge into main
```

---

# 37. Future Improvements

Possible future research/development work includes:

1. Replace the temporal trend forecaster with a supervised future-hazard prediction model after collecting sufficient labeled transition data.
2. Perform formal model evaluation using independent test datasets.
3. Add richer sensor fusion and calibration.
4. Add persistent analytics and historical charts.
5. Add role-based dashboard authentication.
6. Improve camera/person tracking under difficult lighting and occlusion.
7. Add structured experiment logging for FYP research results.
8. Upgrade the development/CI Python version after confirming compatibility with the project's ML dependency set.
9. Add deployment automation for a production cloud environment.

---

# 38. Final System Summary

The complete software architecture can be summarized as:

```
                  PREDICTIVE AI HAZARD PREVENTION SYSTEM
                                  |
             +--------------------+--------------------+
             |                                         |
        SENSOR LAYER                              CAMERA LAYER
             |                                         |
       Arduino Reader                         OpenCV / CNN / Face
             |                                         |
       +-----+------+                         +--------+--------+
       |            |                         |                 |
      Gas       Temperature                 Fire           People
       |            |                         |                 |
      RF           LSTM                     CNN        Presence/Recognition
       |            |                         |                 |
       +------------+-------------------------+-----------------+
                                  |
                                  v
                         DECISION ENGINE
                                  |
                       +----------+----------+
                       |                     |
                 Current Risk          Predictive Risk
                       |                     |
                 SAFE/WARNING/          Trend / Forecast /
                  CRITICAL             Early Warning
                       |                     |
                       +----------+----------+
                                  |
                    +-------------+-------------+
                    |             |             |
                  Relay        Firebase      Flask
                 Response       Events       Dashboard
                    |             |             |
                    +-------------+-------------+
                                  |
                           Monitoring & Alerts
```

---

## 39. Quick Start

For the fastest software-only demonstration:

```bash
git clone https://github.com/cloudforge-creator/fyp-hazard-prevention-system.git
cd fyp-hazard-prevention-system

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python src/main.py --sim --no-firebase
```

Open:

```
http://localhost:5000
```

Run automated tests in another terminal:

```bash
python -m pytest -q
```

Build the container:

```bash
docker build -t fyp-hazard-prevention .
```

---

## 40. Documentation

Detailed software validation information is available in:

```
docs/SOFTWARE_VALIDATION.md
```

This README describes the project from initial setup and architecture through AI processing, integration, simulation, testing, Dockerization, CI, and the final software scope.

---

## License

This repository is an academic Final Year Project. Add an explicit open-source license if the project is intended to be reused or redistributed under specific terms.
