# Project Health and Release-Readiness Audit

**Audit date:** 2026-10-16  
**Repository:** `cloudforge-creator/fyp-hazard-prevention-system`  
**Reviewed branch snapshot:** `main` at `d688ff3fdf265e5b1a27ee040524c67c15852768`  
**Audit type:** Repository/code review, automated-CI review, and review of CNN metrics supplied during this conversation. This is not a formal safety certification or a physical hardware test.

## Executive decision

| Area | Assessment |
|---|---|
| Software simulation/demo | **Suitable for supervised demonstration with known limitations** |
| Automated checks at reviewed commit | **PASS** — 21 unit tests passed; CNN workflow scripts compiled; Docker build passed |
| CNN candidate | **Meets the project's stated 90% test-accuracy target** on the local held-out dataset |
| Live Firebase notifications | **Not configured in the repository snapshot** |
| Physical hardware / relay release | **Not verified** |
| Deployment on a shared network | **Not ready** — dashboard control routes have no authentication/authorization |
| Overall engineering-health estimate | **6.5/10** (heuristic assessment, not a standardized score) |
| Hardware/production release readiness | **3/10** until the critical findings below are resolved and validated |

The software repository is in a useful prototype/demo state. A green CI run proves the tested code passes those automated checks; it does not prove real-world fire detection, sensor calibration, relay behavior, notification delivery, or operational safety.

## Evidence reviewed

- GitHub Actions on the reviewed `main` commit: Python unit tests **21 passed, 1 warning**; CNN workflow script compilation passed; Docker build validation passed.
- Existing unit tests cover CNN probability mapping, duplicate-group manifest behavior, decision-engine risk states/overrides, gas rule fallback, and predictive-risk trend/history behavior.
- The local CNN run output provided during this audit reports 3,798 manifest rows: 2,658 training, 570 validation, and 570 held-out test images.
- Candidate CNN results provided during this audit: **90.70% validation accuracy** and **90.3509% test accuracy on 570 images** at threshold 0.50. The reported test confusion matrix is 245 fire correctly detected, 40 missed fire images, 15 false alarms, and 270 no-fire images correctly classified.
- The application code uses `models/cnn_fire_model.h5` in `src/cnn_model.py`. The candidate is written to `models/cnn_fire_model_candidate.h5` locally and has **not** been promoted to the application's configured model path. The candidate weights and local manifests/evaluation reports were not present in the reviewed GitHub tree, so the CNN metrics above are recorded from the run output supplied during review, not independently reproduced by CI.
- The software-validation document describes a previous 150-cycle local simulation run. That is historical simulation evidence; it was not re-run as part of this repository audit.

## Module health scores

Scores are engineering-review estimates based on code organization, visible automated tests, available metrics, and operational readiness. They are not measured accuracy scores or safety certification.

| Module | Score | Evidence and remaining limitation |
|---|---:|---|
| CNN fire classifier and data pipeline | **8/10** | Duplicate-aware manifest and isolated training/evaluation scripts are available; candidate test accuracy is 90.35%. The test still contains 40 missed fires; unseen real-camera scenes have not been validated; candidate is not the model currently configured for runtime. |
| Gas classification | **6/10** | Trained Random Forest/scaler/labels and a rule-based fallback exist; four gas rule tests are present. No current independent gas-model metric or calibrated physical MQ-2 acceptance result is recorded here. |
| LSTM temperature anomaly detection | **6/10** | Model, scaler, threshold and statistical fallback exist. There is no dedicated `tests/test_lstm_model.py` in the repository and no current verified run-specific MAE/RMSE result in this audit. |
| Decision engine and relay commands | **7/10** | Unit coverage exercises SAFE/WARNING/CRITICAL states, flame and CNN overrides, and relay reset. The configured CNN override fires on a single reading at/above 0.80, so false-positive behavior and relay operation need hardware-domain validation. |
| Predictive-risk / early-warning logic | **7.5/10** | Trend direction, insufficient history, risk preservation, and reset behavior have unit tests. This is a short-horizon trend forecaster, not a validated predictor of exact future fire events. |
| Main application integration and simulation | **7/10** | Integrated sensor loop and documented previous simulation demonstrate the software path. Missing sensor fields can be replaced with placeholder values, and the serial-reader error path can return simulated values even when hardware mode was requested. |
| Arduino sensor acquisition and relay hardware path | **4/10** | Serial protocol and sketch exist, and CI validates Python packaging only. Clean-air calibration uses a fixed starting value; physical sensor calibration, relay polarity, hardware failure cases, and load behavior have not been evidenced in this audit. |
| Person recognition and presence | **5/10** | Haar cascade + LBPH recognition, registration, and time-based presence expiry are implemented. No dedicated tests are present, matching score is not a calibrated identity probability, and local biometric data requires restricted access and retention controls. |
| Flask dashboard / API | **4/10** | Dashboard, live-data, events, video, health, relay reset, and registration routes exist. It binds to `0.0.0.0`; there is no authentication/authorization on the relay-reset and person-registration routes, and the camera stream is accessible without a login. |
| Firebase database and push notifications | **5/10** | Database/event/FCM code and simulation behavior are present. The repository still has a placeholder database URL; real service-account credentials and device-token configuration are intentionally not committed. Live cloud/FCM delivery remains unverified. |
| Automated testing / CI / Docker | **9/10** | Current main-branch CI passed 21 tests, compilation checks for the new CNN workflow scripts, and Docker build validation. Test coverage does not include physical hardware, live Firebase, dashboard access-control, or all ML modules; no coverage percentage is reported. |
| Documentation and operational guidance | **8/10** | README, hardware bring-up checklist, CNN workflow, and software validation report exist. The software validation report's cited commit was stale at the time of this audit and is updated alongside this document. |

## Findings requiring action before a hardware or shared-network release

### H1 — Critical: dashboard controls are unauthenticated

**Evidence:** `dashboard/app.py` binds Flask to `0.0.0.0`. `POST /reset_relay` invokes `DecisionEngine.reset_relay()`; `POST /persons/register` accepts uploaded images and retrains the local face recognizer. The `/video` route serves camera frames. These routes have no application-level authentication/authorization.

**Risk:** Any reachable client on the bound network may be able to invoke a relay reset (which can restore machine power), register/retrain identities, or view camera output.

**Release gate:** Do not expose the dashboard to an untrusted/shared LAN or the internet until access control, authorization for relay operations, CSRF protection for browser-originated state-changing requests, and appropriate network restrictions have been reviewed and tested.

### H2 — Critical: hardware communication failures can be masked as simulated sensor readings

**Evidence:** `src/arduino_reader.py` returns `self._simulate()` on serial read exceptions, even when the instance was created for hardware mode. `src/main.py` also substitutes `5.0` for a missing gas ratio and `25.0` for a missing temperature, and the reader initializes flame state to `0` when no flame reading has been parsed.

**Risk:** Faulty, missing, or malformed sensor input can be indistinguishable from ordinary readings. This is not an acceptable assumption for a real hazard-prevention release.

**Release gate:** Explicit simulation must remain limited to requested simulation mode. Define and test a sensor-fault state that does not silently use synthetic readings in hardware mode, including how the system handles partial samples, stale flame readings, serial disconnects, and relay state during a fault. This needs dedicated unit tests plus physical fault-injection tests before hardware use.

### H3 — Critical: physical safety behavior has not been validated

**Evidence:** The Arduino sketch uses a fixed `CLEAN_AIR_VALUE` starting value and drives the relay output HIGH for `RELAY:ON` and LOW for `RELAY:OFF`; hardware setup documentation explicitly requires calibration and acceptance records. No physical calibration/relay acceptance result was present in the reviewed evidence.

**Risk:** Sensor ratios, active-low/active-high relay module wiring, loss-of-power behavior, and actual load isolation cannot be established by simulation or Docker CI.

**Release gate:** Complete and record the hardware bring-up checklist, safe relay test under supervised conditions, calibration procedure, missing-sensor/disconnect tests, and observed response latency. Do not claim 100% alert activation or hardware accuracy without results.

### H4 — High: candidate CNN still misses a meaningful number of fire samples

**Evidence:** On the supplied held-out test result, fire recall is **85.96%** (245/285 fire samples detected; 40 missed). Overall test accuracy is 90.35%.

**Risk:** Overall accuracy can hide missed fire events. The model result is a dataset score, not a guarantee for live camera conditions.

**Release gate:** Record the 90.35% test result and class-wise metrics truthfully. Before real safety use, evaluate representative, independently labeled camera footage and review false negatives. Do not tune the model or threshold against the already-used test set. The candidate should not replace `models/cnn_fire_model.h5` solely because it crossed the overall 90% target.

### H5 — High: real Firebase operation is unconfigured in the repository snapshot

**Evidence:** `src/firebase_handler.py` contains the placeholder `https://YOUR-PROJECT-rtdb.firebaseio.com`; a real service-account file is not committed by design. `dashboard/templates/register_token.html` contains placeholder web Firebase configuration and VAPID key values.

**Risk:** The application's simulation path can function while actual live database writes or push-notification delivery remain unavailable.

**Release gate:** Configure deployment-specific values outside source control; verify a real live-sensor update, event log, admin device token registration, push delivery, and documented fallback behavior. Never commit service-account files or private keys.

### H6 — High: Python 3.10 is end-of-life

**Evidence:** The current environment emits the `google.api_core` Python-version warning. Python.org identifies Python 3.10.22 as the final Python 3.10 release and states that the series no longer receives security updates: https://www.python.org/downloads/release/python-31022/

**Risk:** Continued use creates maintenance and security exposure, while changing the pinned ML environment without compatibility testing can break TensorFlow dependencies.

**Release gate:** Make a separate maintenance change to a supported Python version after validating the full pinned ML stack, clean-install procedure, model load compatibility, unit tests, and Docker build.

## Recommended order of work

1. **Do not deploy outside an isolated development/demo environment** until H1 and H2 are addressed and tested.
2. Correct sensor-fault behavior so hardware failures are never represented as synthetic readings, then add tests for disconnects and partial/malformed samples.
3. Restrict dangerous dashboard actions and camera/person routes to authorized operators; verify controls cannot be invoked by an unauthenticated client.
4. Complete the supervised physical hardware, sensor calibration, relay, and fault-injection acceptance checklist.
5. Configure and verify real Firebase/FCM in a deployment environment without committing secrets.
6. Keep the current CNN result documented; separately validate the candidate on representative real camera footage before any production-model replacement.
7. Plan the Python runtime migration and expand unit coverage for LSTM, dashboard, Firebase, person recognition, and sensor-reader failure paths.

## Release decision

**Software-only supervised demo:** conditionally suitable, with the current CI status and the documented limitations.

**Hardware-connected or shared-network release:** **not approved by this audit** until H1–H3 are closed and their validation evidence is recorded. H4–H6 also remain important model-quality, cloud-configuration, and maintenance gates.

No application behavior is changed by this documentation audit. The scores and findings are a project-health assessment, not proof that the system is safe for unattended operation.
