# Software Validation Report

## Scope

This report records software-level validation for the Predictive AI Hazard Prevention System. Hardware installation, sensor calibration, physical relay testing, and physical camera validation are outside the scope of the automated CI jobs.

For the broader module scores, risks, and release decision, see [Project Health Audit — 2026-10-16](PROJECT_HEALTH_AUDIT_2026-10-16.md).

## Automated CI Validation

GitHub Actions workflow: `.github/workflows/software-validation.yml`

Reviewed source snapshot: `main` at commit `d688ff3fdf265e5b1a27ee040524c67c15852768` (2026-10-16). The post-merge workflow run passed: [GitHub Actions run](https://github.com/cloudforge-creator/fyp-hazard-prevention-system/actions/runs/38865487438).

- Python unit tests: **PASS — 21 passed, 1 warning**.
- Manifest-builder, manifest-training, and manifest-evaluation scripts: **compiled successfully**.
- Docker build validation: **PASS**.
- The warning is from `google.api_core` about Python 3.10 support; it did not fail the checks.
- CI validates software tests and container build. It does not exercise connected hardware, calibrated sensors, live Firebase credentials, physical relay switching, or camera performance on unseen environments.

## Previously Recorded Local Simulation / End-to-End Validation

The repository previously recorded a 150-cycle local simulation using:

```powershell
python src/main.py --sim
```

The recorded observations were:

- Application initialized components in simulation mode.
- Random Forest gas model, LSTM temperature model, and MobileNetV2 CNN loaded.
- Flask dashboard started on port 5000.
- Requests to `/`, `/data`, `/events`, and `/video` returned successfully during the run.
- Predictive WARNING and a simulated CRITICAL state were observed.
- Simulated relay activation, Firebase notification, and hazard event logging were observed.
- The system subsequently returned to SAFE after the simulated fire signal cleared.

**Provenance note:** This is historical simulation evidence recorded in the project documentation; it was not re-executed by the CI run above or by the 2026-10-16 repository audit. Simulation output does not validate physical sensor accuracy, relay behavior, or real Firebase delivery.

## Software Components Covered by Existing Tests and CI

1. Application integration and simulation pathways (previously recorded local run)
2. Gas-model rule fallback
3. CNN fire-probability mapping
4. Decision-engine classifications and emergency override
5. Predictive-risk trend/history behavior
6. Duplicate-group manifest safeguards
7. Python compilation for the manifest-aware CNN training/evaluation workflow
8. Docker image build
9. GitHub Actions validation

The automated test suite currently contains 21 tests. There is no reported coverage percentage, and automated unit/CI checks do not replace integration and physical acceptance tests for LSTM inference, dashboard access control, person recognition, Firebase, sensor disconnect behavior, camera-domain performance, or relay hardware.

## Known Environment Note

The development environment and CI currently use Python 3.10. Python.org identifies Python 3.10.22 as the final Python 3.10 release and states that the series no longer receives security updates: https://www.python.org/downloads/release/python-31022/

The pinned ML dependencies, including TensorFlow 2.13.0 and NumPy 1.24.3, must be tested together before migrating to a supported Python release. Do not upgrade NumPy independently of the TensorFlow stack.

## Completion Statement

For the defined **software-only supervised demo scope**, core integration, automated tests, Docker build validation, and a previously recorded local simulation have been demonstrated.

This report does **not** certify the system for unattended, hardware-connected fire safety. Review the project health audit and close its critical dashboard-access and sensor-fault findings, then document physical calibration, relay, notification, and real-camera validation before a hardware or shared-network release.
