# CNN webcam false-positive audit

This diagnostic uses the existing `models/cnn_fire_model.h5` without changing weights, inference thresholds, or the application decision engine. It intentionally measures the current model on a labeled webcam scene before any corrective action is considered.

## Run a no-fire baseline

From the repository root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python tools\cnn_webcam_diagnostic.py --label no_fire --seconds 60
```

Keep the webcam pointed at a normal, representative room scene for the entire capture. The script writes a timestamped directory under `artifacts/cnn_webcam_audit/` with:

- `predictions.csv`: full-frame, center-crop, combined, and median-filtered probabilities for every frame.
- `summary.json`: frame-level accuracy for this labeled session, fraction predicted as fire, median probability, mean probability, and 95th percentile.
- `frames/`: sample raw frames saved every five seconds by default.

To disable snapshots, add `--snapshot-every 0`.

## Optional positive check

Only use a clearly labeled fire image/video on a separate screen; never light a real fire:

```powershell
python tools\cnn_webcam_diagnostic.py --label fire --seconds 30
```

This positive check is still only a scene-specific probe. A reliable evaluation needs many independent, labeled scenes and should include both fire and no-fire cases under varied lighting, backgrounds, distances, and camera positions.

## Interpretation and safety

- A high fire-prediction fraction during a no-fire session confirms a scene-specific false-positive issue.
- Compare `full_frame_fire_probability` with `center_crop_fire_probability` to see whether the crop path is disproportionately high.
- This tool uses the project's existing class mapping (`fire=0`, `no_fire=1`), MobileNetV2 preprocessing, full-frame/center-crop weights (0.65/0.35), and a three-frame median.
- Frames from one continuous video are correlated; frame-level accuracy is not equivalent to independent-image accuracy.
- The script does not retrain or alter the model. Do not claim 100% accuracy or safety certification from this report.
