# Predictive AI Hazard Prevention System

**Final Year Project (FYP) — software implementation and software-only demonstration**

This repository contains the software for a predictive hazard-monitoring prototype that combines gas classification, temperature anomaly detection, camera-based fire detection, risk fusion, short-horizon trend forecasting, person presence/recognition, event handling, a Flask dashboard, and simulation/testing infrastructure.

**Software-scope status:** The planned software modules and their integration are documented as the project's software deliverable. Hardware commissioning and laboratory validation are separate activities. This status does **not** mean the repository is production-certified, that every optional external service is configured, or that model accuracy has been independently certified.

## Project team

- Muhammad Ahad
- Muhammad Wajdan
- Muqaddar Hayat

**Supervisor:** Mr. Hamid Nawaz  
**Department:** Software Engineering, NUML Faisalabad Campus

---

## 1. Project objectives

The system is designed to help monitor potentially hazardous conditions and raise early warnings by combining multiple software signals. It is an academic prototype, not a replacement for certified industrial fire, gas, or life-safety equipment.

The software objectives are to:

- process sensor readings and camera frames;
- classify gas-risk categories with a Random Forest model;
- identify temperature anomalies using an LSTM forecasting model;
- estimate fire probability using a MobileNetV2-based image classifier;
- combine signals into SAFE, WARNING, and CRITICAL states;
- forecast short-term risk trends for early warning;
- associate visible people with hazard events where recognition data is configured;
- provide simulated relay actions, event logging, and optional Firebase notifications;
- expose monitoring data through a Flask dashboard and API;
- support local simulation, automated tests, CI, and Docker image building.

## 2. System architecture

```text
Gas readings ──────> Random Forest ───┐
Temperature series ─> LSTM anomaly ───┤
Camera frames ──────> CNN fire score ─┤
Flame signal ─────────────────────────┤
                                      v
                            Decision Engine
                         SAFE / WARNING / CRITICAL
                                      |
                           Predictive Risk Trend
                                      |
                 +--------------------+-------------------+
                 |                    |                   |
           Relay interface       Event handling      Flask dashboard
                 |                    |                   |
          Simulation output     Optional Firebase     REST endpoints
```

The principal software path is:

1. Acquire readings from simulation or the serial-reader abstraction.
2. Run the available model/rule inference paths.
3. combine the available risk signals in the Decision Engine.
4. calculate a short-horizon risk trend.
5. publish state and events to the dashboard/API.
6. invoke simulated or configured response paths when thresholds are crossed.

The actual behavior depends on the model files, local configuration, camera/serial availability, and Firebase setup.

## 3. Technology stack

| Area | Technology |
|---|---|
| Language | Python |
| Gas classification | scikit-learn Random Forest |
| Temperature anomaly forecasting | TensorFlow/Keras LSTM |
| Fire image classification | TensorFlow/Keras MobileNetV2 |
| Image and face processing | OpenCV |
| Web application/API | Flask |
| Model/data utilities | NumPy, pandas, joblib, scikit-learn |
| Optional cloud events and notifications | Firebase Admin SDK / Firebase web configuration |
| Hardware serial interface | pySerial |
| Tests | pytest |
| CI | GitHub Actions |
| Containerization | Docker |

Dependency versions are pinned in `requirements.txt`. The current dependency set targets Python 3.10; install it in an isolated virtual environment.

## 4. Repository layout

```text
.
├── .github/workflows/software-validation.yml
├── arduino/
│   └── hazard_sensors.ino
├── dashboard/
│   ├── app.py
│   ├── static/js/firebase-sw.js
│   └── templates/
├── data/
│   ├── gas_dataset/
│   └── temperature_dataset/
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
│   └── CNN_TRAINING_WORKFLOW.md
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
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

**Artifact availability note:** The gas and temperature artifacts listed above are tracked in the current repository. The current `main` branch does not contain `models/cnn_fire_model.h5`, and the CNN image dataset is not tracked under `data/`. The CNN training/evaluation source is present; the CNN model must be generated from the local dataset before the trained CNN inference path can load a model. Do not describe a missing model artifact as already deployed.

## 5. Software modules

### 5.1 Data acquisition and simulation

Relevant files: `src/arduino_reader.py`, `src/main.py`, `arduino/hazard_sensors.ino`.

The application provides a serial-reader abstraction and simulation mode. Simulation is intended for software demonstration without an attached Arduino. Simulated readings are not evidence of physical sensor accuracy or hardware response.

### 5.2 Gas classification — Random Forest

Relevant files: `src/gas_model.py`, `notebooks/01_train_gas_model.py`.

The model pipeline uses engineered features from MQ-2-style readings and classifies the configured gas-risk categories. The included training script currently generates a synthetic training set from assumed sensor profiles; its evaluation results therefore do not establish accuracy on independently collected, calibrated physical-sensor readings. The tracked gas model, scaler, and label mapping are available in `models/`.

### 5.3 Temperature anomaly detection — LSTM

Relevant files: `src/lstm_model.py`, `notebooks/02_train_lstm_model.py`.

The LSTM uses a rolling sequence of historical readings to estimate the next value and compares prediction error against a threshold. The temperature dataset is tracked at `data/temperature_dataset/machine_temperature_system_failure.csv`. The model, scaler, and threshold artifacts are tracked in `models/`. This dataset is a machine-temperature time series, not a calibrated DHT22 laboratory dataset.

### 5.4 Camera fire detection — MobileNetV2 CNN

Relevant files: `src/cnn_model.py`, `notebooks/03_train_cnn_model.py`, and the manifest-aware scripts `notebooks/06_*` through `notebooks/09_*`.

The runtime expects a Keras model at:

```text
models/cnn_fire_model.h5
```

The runtime resizes frames to 224 × 224 and uses MobileNetV2 preprocessing. The training workflow expects two image classes under `data/fire_dataset/fire/` and `data/fire_dataset/no_fire/`. The current public repository does not include those image folders or the runtime model file. The dataset must be available locally and the selected model must be trained/evaluated and placed at the expected path before CNN inference can use it.

For the duplicate-aware, leakage-conscious workflow, follow [CNN_TRAINING_WORKFLOW.md](notebooks/CNN_TRAINING_WORKFLOW.md). Keep the held-out test split separate from model selection. A validation-accuracy target is not a guaranteed result or a safety certification.

The camera module also contains visual signal processing and temporal filtering. Such heuristics are secondary signals and are not calibrated fire probabilities.

**Model compatibility warning:** The current runtime loader expects a Keras image-classification model. YOLO/object-detection weights are a different model format and output contract; they cannot be substituted directly without adapting and validating the inference integration.

### 5.5 Person presence and recognition

Relevant file: `src/person_recognition.py`.

The camera pipeline supports face detection/recognition using OpenCV components and can associate recognized or unknown visible people with an event. Recognition depends on suitable local registration data and camera conditions. It identifies people visible in the camera view; it does not calculate exact physical distance to a hazard. Biometric registration data should not be committed publicly.

### 5.6 Decision Engine

Relevant file: `src/decision_engine.py`.

The engine combines available gas, temperature, and camera/fire signals into the configured risk state. The documented conceptual weights are:

- CNN fire signal: 0.50
- gas risk: 0.35
- temperature anomaly: 0.15

The implementation also includes a high-risk gas adjustment and an immediate critical override for configured flame/fire conditions. These are application decision rules, not a claim that the fused score is a statistically calibrated probability of an accident.

### 5.7 Predictive risk / early warning

Relevant file: `src/predictive_risk.py`.

The component keeps a rolling history of risk signals, estimates trends with least-squares regression, projects the trend over a short horizon, and classifies the projected risk. It should be described as **short-horizon predictive risk / early warning**, not exact prediction of the time or certainty of a future fire.

### 5.8 Event handling and response interface

Relevant files: `src/predictive_logger.py`, `src/firebase_handler.py`, `src/arduino_reader.py`.

The software can log hazard events and request relay state changes. In simulation mode, relay activation is represented as a simulated action; it does not cut power to a physical machine. Firebase notifications and cloud event storage require valid project configuration and credentials.

### 5.9 Dashboard and API

Relevant file: `dashboard/app.py`.

The Flask dashboard provides a live monitoring view and API routes for sensor/model data, events, video, person registration, and relay reset. The application is intended for a controlled development/demo environment. Do not expose it to a public or shared network without authentication, authorization, and appropriate endpoint protection; the documented implementation has routes that require security hardening before such exposure.

## 6. Data and model artifact inventory

| Asset | Current repository status | Notes |
|---|---|---|
| Gas dataset files | Tracked | `data/gas_dataset/batch1.dat` through `batch10.dat` |
| Gas RF model/scaler/labels | Tracked | Synthetic training assumptions still apply |
| Temperature dataset | Tracked | Machine temperature time series |
| LSTM model/scaler/threshold | Tracked | Saved inference artifacts |
| CNN training/evaluation scripts | Tracked | Multiple workflows available |
| CNN image dataset | Not tracked | Supply local class folders |
| Runtime CNN model `cnn_fire_model.h5` | Not present on current main | Train/evaluate compatible model before CNN inference |
| Person registration/biometric data | Local/private data | Not committed |
| Firebase service-account key | Not tracked | Configure privately if needed |

Never commit service-account keys, API secrets, private face data, or other credentials to Git.

## 7. Install and run

### 7.1 Clone and create an environment

Windows PowerShell:

```powershell
git clone https://github.com/cloudforge-creator/fyp-hazard-prevention-system.git
Set-Location .\fyp-hazard-prevention-system

py -3.10 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Python 3.10 is not installed, install the Python version required by the pinned dependency set before creating the environment. Do not upgrade NumPy independently of TensorFlow; review the full dependency set together.

### 7.2 Run software simulation

```powershell
python src/main.py --sim --no-firebase
```

Open [http://localhost:5000](http://localhost:5000).

Simulation mode is useful for demonstrating application flow without Arduino hardware. Because the CNN runtime artifact is absent from the current repository snapshot, the CNN path may report that its trained model is unavailable until a compatible model has been trained and installed. Do not interpret a simulation-only run as validation of trained camera inference.

Other options documented by the application include:

```powershell
python src/main.py --sim
python src/main.py --port COM4
python src/main.py --no-firebase
python src/main.py --sim --web-port 8000
```

Use `--no-firebase` when cloud credentials are not configured.

## 8. Train and evaluate models

Run commands from the repository root in the activated environment.

### Gas Random Forest

```powershell
python notebooks/01_train_gas_model.py
```

This script trains from its configured synthetic MQ-2 profile generator and saves artifacts under `models/`. Treat its metrics as synthetic-data evaluation, not independent sensor validation.

### Temperature LSTM

```powershell
python notebooks/02_train_lstm_model.py
```

The script reads the tracked temperature dataset and saves the model/scaler/threshold artifacts under `models/`.

### CNN fire classifier

First supply the local dataset:

```text
data/
└── fire_dataset/
    ├── fire/
    └── no_fire/
```

For the duplicate-aware workflow:

```powershell
python notebooks/06_build_cnn_split_manifest.py
python notebooks/07_train_cnn_manifest.py
python notebooks/08_evaluate_cnn_manifest.py --model models/cnn_fire_model_candidate.h5
```

Review [notebooks/CNN_TRAINING_WORKFLOW.md](notebooks/CNN_TRAINING_WORKFLOW.md) before running these commands. The workflow writes candidate artifacts and evaluation outputs; the runtime's expected filename is `models/cnn_fire_model.h5`. Only promote a selected compatible candidate after evaluation and verification. Do not overwrite the runtime model merely because training completed.

## 9. Tests, CI, and Docker

Run unit tests:

```powershell
python -m pytest -q
```

Compile the manifest-aware CNN workflow scripts:

```powershell
python -m py_compile notebooks/06_build_cnn_split_manifest.py
python -m py_compile notebooks/07_train_cnn_manifest.py
python -m py_compile notebooks/08_evaluate_cnn_manifest.py
python -m py_compile notebooks/09_evaluate_cnn_unseen_photos.py
```

Build the Docker image:

```powershell
docker build -t fyp-hazard-prevention .
```

The GitHub Actions workflow at `.github/workflows/software-validation.yml` installs dependencies, compiles CNN workflow scripts, runs pytest, and builds the simulation image. CI passing means those configured checks passed for that commit; it does not prove physical sensor behavior, live Firebase delivery, production security, or accuracy on unseen camera footage.

## 10. Validation evidence and interpretation

The repository includes [docs/SOFTWARE_VALIDATION.md](docs/SOFTWARE_VALIDATION.md) and [docs/PROJECT_HEALTH_AUDIT_2026-10-16.md](docs/PROJECT_HEALTH_AUDIT_2026-10-16.md).

These documents record historical checks for a previously reviewed commit, including a documented 150-cycle simulation and a prior CI run. They are historical evidence, not a claim that the exact current commit has just been rerun. Always use the commit SHA and associated Actions run when reporting validation.

Report model quality using an appropriate held-out labeled test set and metrics such as precision, recall, F1, confusion matrix, and (where applicable) ROC-AUC. Runtime success alone does not establish model accuracy. Do not present a target accuracy as an achieved result.

## 11. Firebase configuration

Firebase is optional for the local software-only demo. To enable it, configure the appropriate private service-account credentials, Realtime Database URL, web configuration, and FCM/VAPID settings required by the implementation. Keep all private credentials out of source control.

Without valid credentials, use:

```powershell
python src/main.py --sim --no-firebase
```

A code path for notification delivery does not prove that a real device received a notification. Verify actual delivery separately when Firebase is configured.

## 12. Arduino interface (hardware integration)

The Arduino sketch is at `arduino/hazard_sensors.ino`. The intended serial protocol uses 9600 baud and the documented pin mapping is:

| Component | Pin |
|---|---|
| MQ-2 gas sensor | A0 |
| DHT22 temperature/humidity sensor | D2 |
| Flame sensor | D4 |
| Relay module | D7 |

Use `python src/main.py --port COM4` with the correct port when the physical board is connected. Check [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) for wiring and setup guidance.

Physical calibration, sensor accuracy, electrical safety, relay switching, and camera placement are outside software-only validation. Never connect a live machine or rely on this academic prototype for real-life safety without appropriate engineering review and independent safety testing.

## 13. Scope and completion statement

The project software scope comprises the application modules, model-training/inference code, decision logic, predictive-risk logic, dashboard/API, optional Firebase integration, simulation mode, test suite, CI workflow, Docker configuration, and technical documentation present in this repository.

For academic project reporting, distinguish these separate dimensions:

- **Software implementation:** source modules and integration included in the defined software-only prototype scope.
- **Reproducibility of trained inference:** depends on the required dataset and model artifacts being available; the CNN runtime model and image dataset are not present in the current public main branch.
- **Software verification:** limited to the specific tests, CI runs, and historical simulation evidence recorded for their respective commits.
- **External configuration:** Firebase credentials, local person-registration data, camera availability, and serial-device configuration are environment-specific.
- **Physical commissioning:** hardware calibration and physical response testing are separate from software implementation.

This repository is an academic prototype. It is not a certified fire-protection product and must not be represented as production-ready or as guaranteeing prevention of hazards.

## 14. Troubleshooting

**PowerShell cannot activate the environment**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**Port 5000 is occupied**

```powershell
python src/main.py --sim --no-firebase --web-port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

**CNN model is not found**

Confirm the local dataset is arranged under `data/fire_dataset/fire/` and `data/fire_dataset/no_fire/`, follow the CNN training workflow, evaluate a candidate, and place the selected compatible Keras model at `models/cnn_fire_model.h5`.

**Firebase errors**

Check private credentials and project configuration, or use `--no-firebase` for a local simulation.

**Tests fail**

Run `python -m pytest -q`, identify the failing test, and review the error before changing implementation or dependency versions.

## 15. Development workflow

```text
Create branch
    ↓
Make a focused change
    ↓
Run relevant tests
    ↓
Run the software simulation
    ↓
Commit and push
    ↓
Open pull request
    ↓
Review CI results
    ↓
Merge after review
```

## 16. Key documentation

- [Software validation report](docs/SOFTWARE_VALIDATION.md)
- [Project health audit](docs/PROJECT_HEALTH_AUDIT_2026-10-16.md)
- [Hardware setup](docs/HARDWARE_SETUP.md)
- [CNN training workflow](notebooks/CNN_TRAINING_WORKFLOW.md)

## License

This is an academic Final Year Project. No open-source license is currently declared. Reuse and redistribution permissions should be specified by the project owners before public reuse.
