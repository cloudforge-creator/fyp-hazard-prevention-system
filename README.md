# Predictive AI Hazard Prevention System

**Final Year Project (FYP) | Software architecture, implementation guide, operation, training, verification, and handover**

The Predictive AI Hazard Prevention System is an academic software prototype for monitoring potentially hazardous conditions. It combines gas-risk classification, temperature anomaly detection, camera-based fire classification, multi-signal risk fusion, short-horizon predictive risk assessment, person presence/recognition, event logging, a Flask dashboard, optional Firebase integration, and simulated or serial-based response interfaces.

> **Scope and safety:** This is an academic prototype, not a certified life-safety product. Use simulation for software demonstrations. Do not rely on it as the sole protection against fire, gas, injury, or equipment damage. Hardware calibration, controlled physical tests, independent safety review, and deployment security are separate acceptance activities.

## Project information

| Field | Details |
|---|---|
| Project | Predictive AI Hazard Prevention System |
| Type | Final Year Project |
| Team | Muhammad Ahad, Muhammad Wajdan, Muqaddar Hayat |
| Supervisor | Mr. Hamid Nawaz |
| Department | Software Engineering, NUML Faisalabad Campus |
| Primary language | Python |
| Web interface | Flask |
| Repository | [cloudforge-creator/fyp-hazard-prevention-system](https://github.com/cloudforge-creator/fyp-hazard-prevention-system) |

## Contents

1. [Purpose and objectives](#1-purpose-and-objectives)
2. [System architecture](#2-system-architecture)
3. [End-to-end processing flow](#3-end-to-end-processing-flow)
4. [Software modules](#4-software-modules)
5. [Risk fusion and prediction logic](#5-risk-fusion-and-prediction-logic)
6. [Repository structure](#6-repository-structure)
7. [Datasets and model artifacts](#7-datasets-and-model-artifacts)
8. [Requirements and installation](#8-requirements-and-installation)
9. [Run the application](#9-run-the-application)
10. [Train and evaluate the models](#10-train-and-evaluate-the-models)
11. [Dashboard and API reference](#11-dashboard-and-api-reference)
12. [Firebase and notification configuration](#12-firebase-and-notification-configuration)
13. [Person registration and privacy](#13-person-registration-and-privacy)
14. [Testing and continuous integration](#14-testing-and-continuous-integration)
15. [Docker](#15-docker)
16. [Arduino and physical integration](#16-arduino-and-physical-integration)
17. [Configuration and operational behavior](#17-configuration-and-operational-behavior)
18. [Troubleshooting](#18-troubleshooting)
19. [Security, safety, and limitations](#19-security-safety-and-limitations)
20. [Verification and reporting guidance](#20-verification-and-reporting-guidance)
21. [Development and handover checklist](#21-development-and-handover-checklist)
22. [Project documentation](#22-project-documentation)

---

## 1. Purpose and objectives

The software is intended to demonstrate a multi-signal approach to hazard monitoring and early warning. Its objectives are to:

- acquire or simulate sensor readings and camera frames;
- classify gas-related risk using a Random Forest model and engineered sensor features;
- forecast temperature readings with an LSTM and flag anomalous prediction errors;
- estimate fire likelihood from camera frames with a MobileNetV2-based CNN;
- combine available signals into SAFE, WARNING, and CRITICAL classifications;
- estimate whether risk is increasing, stable, or decreasing over a short forecast horizon;
- associate camera-visible people with events when registration data is configured;
- log sensor cycles and risk events;
- expose live state, recent events, and video through a Flask dashboard/API;
- optionally write events and live status to Firebase and send configured push notifications;
- provide simulation, tests, CI checks, and a Docker image for a repeatable software demonstration.

### Intended scope

The software scope includes the application entry point, sensor interface, three model paths, decision engine, trend forecaster, event logger, person-recognition component, dashboard, optional cloud integration, training/evaluation scripts, tests, CI, container build, and technical documentation. Hardware commissioning and external-service configuration are environment-specific.

## 2. System architecture

```text
                 +----------------------------+
                 | Data acquisition           |
                 | Arduino serial / simulation|
                 +-------------+--------------+
                               |
              +----------------+----------------+
              |                |                |
              v                v                v
        Gas features     Temperature series   Camera frames
              |                |                |
              v                v                v
        Random Forest        LSTM             MobileNetV2 CNN
              |                |                |
              +----------------+----------------+
                               |
                      +--------v---------+
                      | Decision Engine  |
                      | SAFE / WARNING / |
                      | CRITICAL         |
                      +--------+---------+
                               |
                      +--------v---------+
                      | Predictive Risk  |
                      | Trend / forecast |
                      +--------+---------+
                               |
           +-------------------+-------------------+
           |                   |                   |
           v                   v                   v
     Event + CSV log     Response interface    Flask dashboard
           |             (simulated/serial)    + REST endpoints
           |                   |                   |
           +-------------------+-------------------+
                               |
                     Optional Firebase services
```

The modules run as a Python application. The main loop processes a cycle approximately every 0.5 seconds when processing time permits. Model inference and camera availability can affect the actual cycle time. The dashboard runs in a Flask thread while the sensor loop operates in the background.

## 3. End-to-end processing flow

1. **Start-up:** Parse CLI arguments, select simulation or serial mode, initialize Firebase (or simulation fallback), create the decision engine, and initialize the dashboard integration.
2. **Acquire readings:** Obtain gas raw value, gas ratio, temperature, humidity, and flame state from the Arduino reader or the simulator.
3. **Gas inference:** Use the Random Forest artifact when loadable; otherwise the gas module can fall back to approximate ratio-based rules.
4. **Temperature inference:** Fill the rolling sequence, predict the next temperature with the LSTM when available, and compare prediction error with the saved threshold. A statistical fallback is available when the LSTM artifacts cannot load.
5. **Camera inference:** Read the latest camera frame when enabled, estimate fire probability if a compatible CNN artifact is available, and apply the module's temporal/visual signal handling.
6. **Decision fusion:** Combine gas, temperature, and fire signals, with configured immediate overrides for the flame input or sufficiently high fire signal.
7. **Trend prediction:** Update rolling risk history and estimate the projected risk over the configured horizon.
8. **Publish and record:** Update dashboard state, append a CSV sensor/risk row, update optional cloud state at its configured cadence, and record hazard or predictive-warning events.
9. **Respond:** Execute the configured response path, which may be simulated or serial-command-based depending on the selected mode and attached hardware.
10. **Repeat:** Continue until the application is stopped.

**Important:** fallback behavior helps a demonstration continue when an ML artifact is unavailable, but it does not make that artifact equivalent to a trained model. The UI and evaluation report should distinguish model inference from fallback operation.

## 4. Software modules

| Module | Main files | Responsibility |
|---|---|---|
| Application orchestration | `src/main.py` | CLI, initialization, sensor loop, model orchestration, dashboard hand-off |
| Sensor reader | `src/arduino_reader.py`, `arduino/hazard_sensors.ino` | Serial discovery, parsing, simulated readings, relay commands |
| Gas classification | `src/gas_model.py`, `notebooks/01_train_gas_model.py` | RF inference, feature construction, rule fallback, training |
| Temperature anomaly detection | `src/lstm_model.py`, `notebooks/02_train_lstm_model.py` | Rolling-window prediction, error threshold, statistical fallback |
| Camera fire detection | `src/cnn_model.py`, `notebooks/03_train_cnn_model.py` | Camera capture, CNN inference, fire signal, temporal/visual processing |
| Decision fusion | `src/decision_engine.py` | Weighted score, risk levels, immediate override, response triggers |
| Predictive risk | `src/predictive_risk.py` | Rolling risk history, linear trend estimate, short-horizon projection |
| Data logging | `src/predictive_logger.py` | Append timestamped sensor and prediction values to CSV |
| Person recognition | `src/person_recognition.py` | Haar-cascade face detection, local LBPH recognition, presence tracking |
| Firebase | `src/firebase_handler.py` | Optional database writes, event storage, presence/person records, push-notification paths |
| Dashboard/API | `dashboard/app.py`, `dashboard/templates/` | Live monitoring, events, video, health endpoint, person and relay endpoints |
| Model evaluation | `notebooks/04_evaluate_models.py`, `notebooks/05_*` to `09_*` | Metrics, CNN data quality, split manifest, candidate training, evaluation |
| Automated verification | `tests/`, `.github/workflows/software-validation.yml` | Unit tests, script compilation, CI, Docker build validation |

### 4.1 Sensor reader and simulator

`ArduinoReader` supports automatic serial-port discovery or an explicitly supplied port, uses a 9600-baud serial connection by default, parses the serial key/value stream, and exposes relay activation/deactivation methods. Simulation generates synthetic gas, temperature, humidity, and flame readings for software demonstrations.

When a serial connection cannot be opened, the current reader can switch to simulation mode. A serial read exception can also return simulated values. Therefore, a running application is not sufficient evidence that hardware is connected: check startup logs and the reported operating mode.

### 4.2 Gas model — Random Forest

The gas module uses the gas ratio and raw ADC input to construct the feature vector expected by the trained model. It loads `models/gas_rf_model.pkl`, `models/gas_scaler.pkl`, and `models/gas_labels.pkl` when present. A rule-based ratio fallback is available.

The current training script generates synthetic MQ-2 profiles based on assumed datasheet-style ratio distributions. Metrics produced from that synthetic data are not equivalent to evaluation on independent, calibrated sensor readings or certified gas concentrations. The MQ-2 is a broad-response sensor and does not by itself provide laboratory-grade identification of every listed gas.

### 4.3 Temperature model — LSTM

The temperature model uses a rolling window of 50 readings. When all artifacts load, the LSTM estimates the next scaled temperature, converts it back to the source scale, calculates prediction error, and compares the error with `models/lstm_threshold.pkl`. During initial warm-up it waits for sufficient samples. If the model artifacts are unavailable or fail to load, the implementation uses a rolling statistical anomaly fallback.

The training workflow reads `timestamp` and `value` from the tracked machine-temperature CSV and uses a time-ordered split. This data demonstrates time-series anomaly detection; it is not a substitute for a calibrated target-device temperature dataset.

### 4.4 Camera fire model — MobileNetV2 CNN

The runtime is configured to load `models/cnn_fire_model.h5`. It resizes images to 224 × 224, applies MobileNetV2 preprocessing, and interprets the sigmoid output according to the training class order (`fire=0`, `no_fire=1`), converting the no-fire score into a fire probability. The module also contains center-region inference, temporal smoothing, a visual flame cue, and camera/person integration.

The current public repository tracks the CNN training/evaluation scripts but does not track the required fire-image dataset or the runtime file `models/cnn_fire_model.h5`. To run trained CNN inference from a clean clone, supply the dataset, train and evaluate a compatible Keras model, and install the selected artifact at the configured path. The manifest-aware workflow in [notebooks/CNN_TRAINING_WORKFLOW.md](notebooks/CNN_TRAINING_WORKFLOW.md) is the preferred documented workflow for duplicate-aware splitting.

**Architecture compatibility:** a YOLO object-detection model produces a different output contract and uses a different loader from the current Keras image classifier. YOLO weights cannot be substituted for `cnn_fire_model.h5` without changing and testing the runtime integration.

### 4.5 Decision engine

The decision engine uses the following documented weights for weighted fusion:

| Signal | Weight |
|---|---:|
| CNN fire score | 0.50 |
| Gas risk | 0.35 |
| Temperature anomaly score | 0.15 |
| **Total** | **1.00** |

High-risk gas labels receive an additional configured risk multiplier. The temperature term contributes when the LSTM/statistical path reports an anomaly. The fused score is mapped to SAFE, WARNING, or CRITICAL using the thresholds in `src/decision_engine.py`. A flame-sensor signal or a sufficiently high fire signal can trigger the configured immediate CRITICAL override. The engine also manages response/notification cooldowns.

These are engineered decision rules and weights, not a calibrated probability that a real-world accident will occur. Any threshold changes should be tested against labeled scenarios and documented.

### 4.6 Predictive risk and early warning

The predictive-risk component stores a rolling history of gas, temperature, and fire risk scores. It estimates signal slopes using least-squares regression, projects scores over a configurable forecast horizon, and reports a predicted level, trend, confidence indicator, and early-warning flag.

The method is a short-horizon trend projection. It does not predict an exact time of a future fire and is not a supervised future-event model trained on labeled hazard transitions. The CSV logger can collect timestamped samples for future evaluation and training.

### 4.7 Person presence and recognition

The person module uses OpenCV Haar-cascade face detection and LBPH face recognition. Registration metadata is stored separately from the local LBPH model. It tracks currently visible registered or unknown people and provides information for the dashboard and hazard-event context.

Recognition depends on face image quality, lighting, pose, camera placement, and the quality of registered examples. It is not a general-purpose person detector for every scene and should not be treated as infallible identification. Keep face images, metadata, and recognition models private; do not commit them to a public repository.

### 4.8 Event and CSV logging

The decision engine and dashboard retain recent event information in process memory; the dashboard's local event history is bounded. Firebase event persistence is optional. `PredictiveDataLogger` appends a row to `data/predictive_sensor_log.csv` with a timestamp, gas readings, temperature/humidity, anomaly indicators, fire/flame signals, current risk, predicted risk, and trend.

The CSV is useful for inspection and for building a future labeled dataset. A logged prediction is not a ground-truth label: observed outcomes must be separately labeled before using the data to train or evaluate a future-hazard model.

### 4.9 Dashboard and cloud integration

The Flask application serves the dashboard, JSON endpoints, an MJPEG video stream, health status, person listing/registration, and a relay-reset endpoint. Firebase integration can write live sensor state and event/person/presence data and provide notification-related functionality when configured. The application can be run without cloud credentials in simulation mode.

## 5. Risk fusion and prediction logic

### Current risk classification

The fusion score is calculated from the weighted signal terms in `src/decision_engine.py`, including the gas multiplier and temperature anomaly handling. The source currently defines:

- WARNING threshold: `0.35`
- CRITICAL threshold: `0.65`
- Immediate flame override: flame signal active
- Immediate CNN override: fire probability at or above the configured threshold

The exact behavior is implementation-defined by the code and should be rechecked whenever model output scales or thresholds change.

### Predictive output fields

The decision result can include:

- `level`: current SAFE/WARNING/CRITICAL classification;
- `score`: current fused risk score;
- `predicted_level` and `predicted_score`: projected short-horizon result;
- `prediction_trend`: increasing/stable/decreasing or insufficient-history state;
- `prediction_confidence`: heuristic confidence indicator based on available history;
- `early_warning`: predictive-warning flag;
- `response_level`: response classification, which can indicate a predicted critical state before current risk is critical;
- `current_people`: current camera presence information where available.

The confidence field is a heuristic indicator, not a statistically calibrated confidence interval.

## 6. Repository structure

```text
.
├── .github/
│   └── workflows/software-validation.yml
├── arduino/
│   └── hazard_sensors.ino
├── dashboard/
│   ├── app.py
│   ├── static/js/firebase-sw.js
│   └── templates/
│       ├── index.html
│       └── register_token.html
├── data/
│   ├── gas_dataset/
│   │   ├── batch1.dat ... batch10.dat
│   └── temperature_dataset/
│       └── machine_temperature_system_failure.csv
├── docs/
│   ├── HARDWARE_SETUP.md
│   ├── PROJECT_HEALTH_AUDIT_2026-10-16.md
│   └── SOFTWARE_VALIDATION.md
├── models/
│   ├── gas_labels.pkl
│   ├── gas_rf_model.pkl
│   ├── gas_scaler.pkl
│   ├── lstm_scaler.pkl
│   ├── lstm_temp_model.h5
│   └── lstm_threshold.pkl
├── notebooks/
│   ├── 01_train_gas_model.py
│   ├── 02_train_lstm_model.py
│   ├── 03_train_cnn_model.py
│   ├── 04_evaluate_models.py
│   ├── 05_cnn_quality_audit.py
│   ├── 06_build_cnn_split_manifest.py
│   ├── 07_train_cnn_manifest.py
│   ├── 08_evaluate_cnn_manifest.py
│   ├── 09_evaluate_cnn_unseen_photos.py
│   ├── CNN_TRAINING_WORKFLOW.md
│   └── plots/README.md
├── src/
│   ├── __init__.py
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
├── tests/
│   ├── test_cnn_model.py
│   ├── test_cnn_split_manifest.py
│   ├── test_cnn_status.py
│   ├── test_decision_engine.py
│   ├── test_gas_model.py
│   └── test_predictive_risk.py
├── .dockerignore
├── .gitignore
├── CONTRIBUTING.md
├── Dockerfile
├── README.md
└── requirements.txt
```

This tree describes the tracked project structure. Runtime-generated files such as CSV logs, local CNN data/candidates, Firebase service-account credentials, face images, person metadata, and the LBPH model may be created locally and should be handled according to the privacy and configuration guidance below.

## 7. Datasets and model artifacts

| Data/artifact | Path | Purpose |
|---|---|---|
| Gas source batches | `data/gas_dataset/batch1.dat` … `batch10.dat` | Included gas data files; inspect their format before using them in a training workflow |
| Temperature series | `data/temperature_dataset/machine_temperature_system_failure.csv` | Time-ordered temperature anomaly training series |
| Gas model | `models/gas_rf_model.pkl` | Trained Random Forest artifact |
| Gas scaler | `models/gas_scaler.pkl` | Feature scaling |
| Gas labels | `models/gas_labels.pkl` | Label mapping |
| LSTM model | `models/lstm_temp_model.h5` | Next-reading prediction |
| LSTM scaler | `models/lstm_scaler.pkl` | Temperature scaling |
| LSTM threshold | `models/lstm_threshold.pkl` | Prediction-error threshold |
| CNN dataset | `data/fire_dataset/fire/`, `data/fire_dataset/no_fire/` | Required local image classes for CNN training |
| CNN runtime model | `models/cnn_fire_model.h5` | Filename expected by current CNN runtime |
| CNN candidates and metrics | `models/` and `notebooks/plots/` | Generated by training/evaluation scripts |
| Person data | `data/person_faces/`, `data/persons.json`, `data/person_lbph.yml` | Local biometric registration and model files |
| Prediction log | `data/predictive_sensor_log.csv` | Runtime-generated timestamped data |

The current public `main` branch includes the gas and temperature artifacts shown above, but it does not include the CNN image dataset or the runtime CNN model. Supply the image dataset and produce a compatible model before claiming that the CNN trained-inference path is reproducible from a fresh clone. Generated model weights and reports should be committed only if repository size, licensing, and dataset privacy allow it.

## 8. Requirements and installation

### Requirements

- Windows 10/11, Linux, or another compatible Python environment;
- Python 3.10 for the currently pinned dependency set;
- Git;
- Internet access for initial package installation;
- Optional: Docker Desktop/Engine;
- Optional: Arduino board and sensors for serial integration;
- Optional: webcam for camera inference;
- Optional: Firebase project and private credentials for cloud functionality.

The dependencies are pinned in `requirements.txt`, including TensorFlow 2.13.0 and NumPy 1.24.3. Install the requirements as a compatible set; do not independently upgrade NumPy or TensorFlow without checking compatibility.

### Windows PowerShell setup

```powershell
git clone https://github.com/cloudforge-creator/fyp-hazard-prevention-system.git
Set-Location .\fyp-hazard-prevention-system

py -3.10 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Confirm the environment:

```powershell
python --version
python -m pip --version
python -m pytest --version
```

If PowerShell reports that `Activate.ps1` is missing, verify that the virtual environment was successfully created in the current repository directory before activating it.

### Linux/macOS shell setup

```bash
git clone https://github.com/cloudforge-creator/fyp-hazard-prevention-system.git
cd fyp-hazard-prevention-system

python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 9. Run the application

Run commands from the repository root with the virtual environment activated.

### Simulation without Firebase

```powershell
python src/main.py --sim --no-firebase
```

Open [http://localhost:5000](http://localhost:5000).

Simulation mode does not require Arduino hardware. It uses synthetic readings and simulated response behavior. The current CNN artifact is not included in the public tree, so the camera model path may report an unavailable model until a compatible artifact has been trained and installed.

### Available command-line options

| Argument | Purpose |
|---|---|
| `--sim` | Use simulated sensor values instead of a serial-connected board |
| `--port COM4` | Select a specific serial port (example only; use the actual port) |
| `--no-firebase` | Disable live Firebase use for a local demo |
| `--sim-camera` | Enable the real webcam while sensor values remain simulated |
| `--web-port 8000` | Change the Flask dashboard port |

Examples:

```powershell
python src/main.py --sim
python src/main.py --sim --no-firebase --web-port 8000
python src/main.py --port COM4
python src/main.py --sim --sim-camera --no-firebase
```

Use `Ctrl+C` in the terminal to stop the application. Check the startup logs for the selected mode, model-loading messages, camera availability, and Firebase status. Do not assume hardware is connected merely because the application started.

## 10. Train and evaluate the models

Training commands must be executed from the repository root. Ensure the environment has the dependencies installed and the corresponding input data is present.

### 10.1 Gas Random Forest

```powershell
python notebooks/01_train_gas_model.py
```

The script generates synthetic profiles from configured MQ-2-style ratio assumptions, splits samples, scales features, applies SMOTE to the training portion, trains a Random Forest, and writes model/scaler/label artifacts and plots. Read the script's comments before interpreting the metrics: synthetic-data performance does not demonstrate performance on independent physical sensor data.

### 10.2 Temperature LSTM

```powershell
python notebooks/02_train_lstm_model.py
```

The script reads the temperature CSV, validates the required `timestamp` and `value` columns, sorts the series, creates 50-step sequences, uses a time-ordered validation segment, trains the LSTM, and saves the best model with scaling and anomaly-threshold artifacts. Inspect its generated metrics and plots before replacing tracked artifacts.

### 10.3 CNN fire classifier

Prepare the local dataset in this form:

```text
data/fire_dataset/
├── fire/
│   ├── fire_image_001.jpg
│   └── ...
└── no_fire/
    ├── normal_image_001.jpg
    └── ...
```

The original training script is:

```powershell
python notebooks/03_train_cnn_model.py
```

For the duplicate-aware manifest workflow, use:

```powershell
python notebooks/06_build_cnn_split_manifest.py
python notebooks/07_train_cnn_manifest.py
python notebooks/08_evaluate_cnn_manifest.py --model models/cnn_fire_model_candidate.h5
python notebooks/09_evaluate_cnn_unseen_photos.py --help
```

Review [CNN_TRAINING_WORKFLOW.md](notebooks/CNN_TRAINING_WORKFLOW.md) before training. The manifest-aware workflow groups exact and likely duplicate images so that groups are not spread across training, validation, and test subsets. Inspect any duplicate-label conflicts manually. Select a candidate using validation metrics, then use the held-out test split for final evaluation; do not tune a threshold based on test results.

The candidate model name is not automatically the same as the runtime filename. The current runtime expects `models/cnn_fire_model.h5`. Promote a selected compatible candidate only after checking the evaluation results, class mapping, model input/output shape, preprocessing, and inference integration. A YOLO training log or YOLO weights are not compatible with this Keras loader by filename alone.

### 10.4 General model evaluation

`notebooks/04_evaluate_models.py` supports CNN evaluation and optionally labeled gas/temperature evaluation when supplied with the expected CSV format. CNN evaluation using the simple validation split is validation evidence, not an independent held-out test result. The manifest-aware evaluation scripts should be used for the duplicate-aware candidate workflow.

Recommended metrics for classification include accuracy, per-class precision, recall, F1-score, confusion matrix, and—where appropriate—ROC-AUC. For forecasting/regression, report MAE/RMSE in both scaled and meaningful original units where possible. Record the dataset version, split strategy, model hash, threshold, software environment, and exact command used for every reported result.

## 11. Dashboard and API reference

The Flask server is defined in `dashboard/app.py`. The default port is 5000.

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the main monitoring dashboard |
| `/data` | GET | Return the current in-memory sensor/model/risk state as JSON |
| `/events` | GET | Return recent Firebase events when available, otherwise local event history |
| `/video` | GET | Stream annotated camera frames as multipart MJPEG when frames are available |
| `/health` | GET | Return a simple health status and application uptime |
| `/persons` | GET | Return registered-person metadata when the callback is configured |
| `/persons/register` | POST | Register a person from multipart form fields and image uploads |
| `/reset_relay` | POST | Request a relay reset through the decision engine |

### Person registration request

The registration endpoint expects multipart form data:

- `person_id`: stable local identifier;
- `name`: display name;
- `designation`: role/designation;
- `images`: one or more uploaded face images.

The endpoint depends on the recognition callback being configured. Validate uploads and protect this endpoint before network exposure. Face data is sensitive personal data and must be collected and stored only with appropriate authorization.

### Example: query live data

PowerShell:

```powershell
Invoke-RestMethod http://localhost:5000/data
Invoke-RestMethod http://localhost:5000/events
Invoke-RestMethod http://localhost:5000/health
```

The JSON response fields depend on the current sensor cycle and initialized modules. A `null` fire probability can mean the trained CNN model/camera inference is unavailable; interpret it as unavailable rather than as zero risk.

## 12. Firebase and notification configuration

Firebase is optional for a local simulation. The handler currently expects the database URL in `src/firebase_handler.py` and a private service-account JSON file at the configured path. The browser push-registration page also requires the correct Firebase web configuration and messaging settings.

### Configuration steps

1. Create or select the project in the Firebase console.
2. Enable and configure Realtime Database with appropriate access rules.
3. Configure a service account and store its key privately.
4. Set the correct database URL in the private/local configuration.
5. Configure the web application's Firebase settings and required messaging/VAPID settings.
6. Register a supported client device and grant notification permission.
7. Verify a real test notification end-to-end and review server logs.
8. Keep database rules restrictive and rotate any credential that was accidentally exposed.

The repository's placeholder URL is not a live Firebase project. Without valid configuration, the handler falls back to simulation behavior. A successful local code path or simulated log line is not proof of cloud write success or real device notification delivery.

For local use without Firebase:

```powershell
python src/main.py --sim --no-firebase
```

Never commit `serviceAccountKey.json`, FCM tokens, private keys, or other secrets.

## 13. Person registration and privacy

The recognition component stores its local data under `data/person_faces/`, `data/persons.json`, and `data/person_lbph.yml`. These are runtime/local assets and are not part of the ordinary public source tree.

Operational guidance:

- obtain appropriate consent before enrolling anyone;
- restrict filesystem and endpoint access to face images and recognition files;
- do not upload biometric data to public repositories or issue trackers;
- use strong registration identifiers and avoid exposing personal details in logs;
- provide a process to correct or remove registration data;
- evaluate recognition performance under expected lighting, pose, and camera conditions;
- treat unknown or low-confidence faces as unknown, not as a verified identity.

## 14. Testing and continuous integration

Run the unit tests from the repository root:

```powershell
python -m pytest -q
```

Compile the manifest-aware CNN scripts:

```powershell
python -m py_compile notebooks/06_build_cnn_split_manifest.py
python -m py_compile notebooks/07_train_cnn_manifest.py
python -m py_compile notebooks/08_evaluate_cnn_manifest.py
python -m py_compile notebooks/09_evaluate_cnn_unseen_photos.py
```

The GitHub Actions workflow in `.github/workflows/software-validation.yml` is triggered on pushes to `main` and `chore/**`, and pull requests targeting `main`. It installs the pinned dependencies on Python 3.10, compiles the manifest-aware CNN scripts, runs pytest, and builds the Docker simulation image.

A green workflow proves only the checks included in that workflow passed for that commit. It does not prove that the physical sensors are calibrated, the webcam performs reliably on unseen scenes, Firebase credentials are valid, push notifications reach a device, or the dashboard is safe to expose to a network.

## 15. Docker

Build the simulation image:

```powershell
docker build -t fyp-hazard-prevention .
```

Run it:

```powershell
docker run --rm -p 5000:5000 fyp-hazard-prevention
```

Open [http://localhost:5000](http://localhost:5000).

The Dockerfile uses Python 3.10 slim, installs the pinned requirements and OS libraries needed by OpenCV, copies the application source/dashboard/model artifacts, and starts simulation mode with Firebase disabled. The image is intended for a software demo. It does not automatically include local datasets, local biometric data, Firebase credentials, a webcam, or an attached serial device. Container build success is not an end-to-end model or hardware test.

## 16. Arduino and physical integration

The repository includes `arduino/hazard_sensors.ino` and `docs/HARDWARE_SETUP.md`. The reader defaults to 9600 baud and expects key/value serial fields for gas raw reading, gas ratio, temperature, humidity, and flame state, with an end marker. The exact controller board, pin mapping, sensor revisions, and electrical levels must be confirmed against the actual hardware.

### Safe bring-up order

1. Inspect the exact board and sensor datasheets.
2. Verify supply voltage, logic levels, grounding, analog input limits, and relay polarity.
3. Keep mains and hazardous loads disconnected during initial tests.
4. Start the application in simulation mode and verify the dashboard.
5. Connect the board with the relay load disconnected; select the actual serial port.
6. Verify that every serial field is valid and compare readings with appropriate reference instruments.
7. Test relay commands using a safe low-voltage test load.
8. Perform controlled sensor tests under supervision and record calibration values.
9. Validate the complete response path only after independent electrical and safety review.

Do not rely on assumed pin numbers from an old setup. Follow [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) and confirm the actual board and module specifications before wiring.

## 17. Configuration and operational behavior

| Setting | Current behavior |
|---|---|
| Simulation mode | Synthetic sensor values and simulated relay response |
| Serial mode | Attempts auto-discovery or uses the provided serial port |
| Serial baud rate | 9600 by default |
| Main processing loop | Targets roughly 0.5-second cycles |
| Dashboard port | 5000 by default; configurable via `--web-port` |
| CNN input | 224 × 224 RGB after preprocessing |
| LSTM input | 50 historical readings |
| Risk history | Configured in `src/decision_engine.py` / `src/predictive_risk.py` |
| Firebase | Optional; requires private valid project configuration |
| Person recognition | Local OpenCV Haar + LBPH; requires registered examples |
| Event history | Local in-memory recent history, with optional Firebase-backed retrieval |
| Predictive CSV | `data/predictive_sensor_log.csv` by default |

The implementation contains fallback paths for some missing model or connection cases. Always inspect startup messages and the live data payload to determine whether each model, camera, serial connection, and external service is actually available. A fallback may keep the dashboard running while providing lower-fidelity behavior than the trained model.

## 18. Troubleshooting

### Virtual environment activation fails (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

If the script does not exist, create the environment first with `py -3.10 -m venv venv` from the repository root.

### Dependency installation or TensorFlow import fails

Check that the correct virtual environment is active and Python is the intended version:

```powershell
python --version
python -m pip show tensorflow numpy
```

Use the pinned requirements as a set. Do not independently upgrade NumPy beyond the compatible version required by the pinned TensorFlow package.

### CNN model is not found

Confirm that the dataset is present in `data/fire_dataset/fire/` and `data/fire_dataset/no_fire/`. Follow the manifest-aware training workflow, evaluate the candidate, verify class order and preprocessing, then install the selected compatible model at `models/cnn_fire_model.h5`.

### Camera unavailable

Check OS camera permissions, whether another application is using the camera, and whether OpenCV can open the intended device. A camera-disabled simulation can run without live frames. Do not interpret camera unavailability as evidence that no fire exists.

### Firebase is not connected

Check the configured database URL, private service-account path, database rules, Firebase web configuration, and messaging settings. Use `--no-firebase` for a local demo without cloud credentials.

### Serial board is not detected

Confirm the correct port in Device Manager or the operating system's serial-device listing. Close serial monitor applications that may be holding the port. Check baud rate, wiring, board power, and the expected serial protocol. Verify the log indicates hardware mode rather than fallback simulation.

### Port 5000 is already in use

```powershell
python src/main.py --sim --no-firebase --web-port 8000
```

Open [http://localhost:8000](http://localhost:8000).

### Tests fail

Run `python -m pytest -q`, read the first failing assertion/traceback, and reproduce the failure before changing code. Do not treat a test failure as a reason to delete the test or silently weaken its assertion.

## 19. Security, safety, and limitations

- **Dashboard access:** The current Flask dashboard binds to `0.0.0.0`, and the control/registration routes do not implement authentication and authorization in the reviewed code. Restrict it to a trusted local environment; do not expose it to a public or shared network until access controls and endpoint protections are added and tested.
- **Sensor faults:** Serial connection/read errors can lead to simulation fallback, and some application paths substitute defaults for missing readings. For real deployment, sensor-disconnected, stale, invalid, and out-of-range values must be surfaced as explicit faults rather than silently treated as normal readings.
- **CNN availability:** A missing or unloaded CNN artifact means trained camera inference is unavailable. A heuristic visual cue is not a replacement for a validated classifier.
- **Gas classification:** The gas model's synthetic training assumptions limit what its metrics can establish. Validate with independent data from the intended sensor and controlled concentrations.
- **Temperature anomaly detection:** An anomaly in a public machine-temperature series does not automatically represent a fire condition in the intended physical environment.
- **Person recognition:** Face detection and LBPH recognition are affected by image conditions and should not be treated as infallible identity verification.
- **Predictive risk:** Trend extrapolation is a heuristic, not a guaranteed prediction of a future event or exact incident time.
- **Firebase:** External cloud delivery depends on real configuration, permissions, network connectivity, and client registration.
- **Physical response:** Simulated relay logs do not switch a physical relay. Electrical isolation, relay polarity, load ratings, and safe-state behavior require hardware validation.
- **No certification:** The system has not been demonstrated here to meet any industrial fire alarm, gas detector, or functional-safety standard.

## 20. Verification and reporting guidance

The repository contains [docs/SOFTWARE_VALIDATION.md](docs/SOFTWARE_VALIDATION.md) and [docs/PROJECT_HEALTH_AUDIT_2026-10-16.md](docs/PROJECT_HEALTH_AUDIT_2026-10-16.md). They record checks and observations associated with a previously reviewed commit, including a historical local simulation and an earlier CI run.

When preparing a report or demonstration, identify the exact commit and date for every test result. Do not present historical test output as a fresh test of the current commit. Do not claim an accuracy target was achieved unless the exact model, data split, evaluation script, and metrics support that claim.

A complete verification record should include:

- repository commit SHA and working-tree status;
- Python and dependency versions;
- dataset source, version, license, class counts, and split strategy;
- model artifact names and cryptographic hashes;
- training parameters and random seeds;
- independent evaluation metrics and confusion matrices;
- unit-test output and CI run URL;
- simulation steps and logs;
- Firebase test-delivery evidence if enabled;
- physical calibration and relay-test records when hardware is in scope;
- known limitations and acceptance criteria.

## 21. Development and handover checklist

### Reproducible software handover

- [ ] Clone the repository into a clean directory.
- [ ] Create the Python environment and install the pinned requirements.
- [ ] Record the exact Python version and Git commit SHA.
- [ ] Run pytest and save the test output.
- [ ] Build the Docker simulation image.
- [ ] Start simulation mode and inspect dashboard/API state.
- [ ] Verify the expected gas and LSTM artifacts load.
- [ ] Provide the local CNN dataset and selected compatible CNN model if trained camera inference is required.
- [ ] Confirm Firebase configuration is private and optional services are either configured or intentionally disabled.
- [ ] Keep face registration files and secrets out of Git.
- [ ] Record model metrics with the dataset split and evaluation command.
- [ ] Document any remaining environment-specific assumptions in the project handover.

### Git workflow

```text
Create a focused branch
        ↓
Make one logically scoped change
        ↓
Run relevant tests locally
        ↓
Review the diff and remove secrets/data
        ↓
Commit and push
        ↓
Open a pull request
        ↓
Review CI results
        ↓
Merge after review
```

## 22. Project documentation

- [Software Validation Report](docs/SOFTWARE_VALIDATION.md)
- [Project Health and Release-Readiness Audit](docs/PROJECT_HEALTH_AUDIT_2026-10-16.md)
- [Hardware Setup and Bring-Up Checklist](docs/HARDWARE_SETUP.md)
- [CNN Training Workflow](notebooks/CNN_TRAINING_WORKFLOW.md)
- [Contribution Guidelines](CONTRIBUTING.md)
- [GitHub Actions Workflow](.github/workflows/software-validation.yml)

## License

No open-source license is currently declared in this repository. The project owners should select and add a license before granting public reuse or redistribution rights. Dataset and pretrained-model licenses must also be checked independently.
