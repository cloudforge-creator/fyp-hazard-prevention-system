# Contributing

Thank you for helping improve the Predictive AI Hazard Prevention System.

## Before you start

1. Open an issue for substantial changes or bug reports.
2. Keep changes focused and explain the safety or research motivation.
3. Never commit Firebase service-account keys, real face images, private credentials, or unapproved personal data.
4. Do not describe simulation results as physical-hardware validation.

## Development setup

Use Python 3.10 for the pinned TensorFlow 2.13 environment.

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

If PowerShell blocks activation, use a temporary process-scoped policy or run commands through venv\Scripts\python.exe; do not weaken the machine-wide execution policy just for this project.

## Quality checks

    python -m pytest -q
    python notebooks/04_evaluate_models.py

The evaluation script requires the trained CNN model and dataset by default. Optional gas and temperature evaluations require labeled CSV inputs. Record the dataset source, split, sample count and output metrics.

## Pull requests

- Use a descriptive branch such as fix/sensor-timeout or feat/evaluation-report.
- Include a concise summary, test evidence, and any known limitations.
- Add or update tests for changed behavior.
- Do not commit generated model files or evaluation results unless they are intentionally reviewed and small enough for version control.
- For hardware-facing changes, state whether validation was performed in simulation, on a bench, or on the installed system.

## Safety and privacy

This is an academic prototype, not a certified life-safety device. Do not rely on it as the sole fire alarm or emergency shutdown system. Face recognition data should be collected only with appropriate consent and access controls.
