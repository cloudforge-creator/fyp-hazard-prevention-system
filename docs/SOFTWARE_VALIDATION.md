# Software Validation Report

## Scope

This report covers the software implementation and software-level validation of the Predictive AI Hazard Prevention System. Hardware installation, sensor calibration, physical relay testing, and physical camera validation are outside this software-only scope.

## Automated CI Validation

GitHub Actions workflow: `.github/workflows/software-validation.yml`

Validated on the `main` branch at commit `385c45eae428fc2370f65f209a41567f6ab4fdb1`.

- Python unit-test job: **PASS**
- Docker build validation job: **PASS**
- Dependencies installed successfully in CI.
- The Docker image build completed successfully.

## Local Simulation / End-to-End Validation

Command used:

```powershell
python src/main.py --sim
```

Observed software behavior during a 150-cycle run:

- Application initialized all components successfully.
- Simulation mode operated without hardware.
- Random Forest gas model loaded.
- LSTM temperature model loaded and completed its warm-up period.
- MobileNetV2 CNN fire model loaded.
- Flask dashboard started on port 5000.
- Dashboard requests to `/`, `/data`, `/events`, and `/video` returned successfully during the run.
- Predictive risk processing operated after history accumulation.
- Predictive WARNING states were produced.
- A simulated critical hazard was detected with gas and fire risk.
- Decision Engine raised the CRITICAL state.
- Simulated relay activation was triggered.
- Simulated Firebase push notification was generated.
- Simulated Firebase hazard event logging was generated.
- The system subsequently returned to SAFE operation after the simulated fire signal cleared.

## Software Components Covered

1. Application entry point and sensor-processing loop
2. Simulation input layer
3. Gas Random Forest inference
4. LSTM temperature anomaly inference
5. CNN fire inference
6. Decision Engine and weighted multi-model fusion
7. Predictive risk / early-warning logic
8. Person-presence and hazard-event correlation software
9. Flask dashboard and API routes
10. Firebase integration paths and simulation handlers
11. Relay-control software logic
12. Runtime predictive logging
13. Docker packaging
14. Pytest-based validation
15. GitHub Actions CI validation

## Expected Simulation States

The simulation is expected to show SAFE operation during normal simulated readings, predictive WARNING when projected risk crosses the warning threshold, and CRITICAL when the configured immediate-critical conditions are triggered. Initial predictive output may report insufficient history while the temporal history buffer is warming up.

## Known Environment Note

The current project environment uses Python 3.10. Python 3.10 reaches upstream end-of-life on 2026-10-04. A future maintenance update should migrate the development and CI environment to a supported Python release after confirming compatibility of the pinned ML dependencies.

## Software Completion Statement

For the defined **software-only scope**, implementation, integration, automated CI validation, Docker build validation, and local simulation/end-to-end validation have been completed.

Hardware/physical validation is intentionally excluded from this completion statement.
