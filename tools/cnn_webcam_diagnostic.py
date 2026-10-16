"""Collect a labeled live-webcam audit for the existing CNN without changing the model.

Run from repository root:
  python tools/cnn_webcam_diagnostic.py --label no_fire --seconds 60
  python tools/cnn_webcam_diagnostic.py --label fire --seconds 30

Use --label fire only when a safe, clearly labeled fire image/video is displayed
on a separate screen. Never create a real flame for testing. This script does not
claim real-world accuracy from a single scene; repeat with diverse scenes.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "cnn_fire_model.h5"
SIZE = (224, 224)
ROI_SCALE = 0.70


def fire_probability(model, frame: np.ndarray) -> float:
    """Match src/cnn_model.py preprocessing and fire/no-fire class mapping."""
    resized = cv2.resize(frame, SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(
        rgb.astype(np.float32)
    )[None, ...]
    output = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if output.size != 1:
        raise ValueError(f"Expected one sigmoid output, got shape {output.shape}")
    no_fire_probability = float(np.clip(output[0], 0.0, 1.0))
    return 1.0 - no_fire_probability


def center_crop(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    crop_width = max(1, int(width * ROI_SCALE))
    crop_height = max(1, int(height * ROI_SCALE))
    x1 = max(0, (width - crop_width) // 2)
    y1 = max(0, (height - crop_height) // 2)
    return frame[y1:y1 + crop_height, x1:x1 + crop_width]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("fire", "no_fire"), required=True,
                        help="Ground-truth label for the entire capture session.")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "cnn_webcam_audit")
    parser.add_argument("--snapshot-every", type=int, default=5,
                        help="Save a raw frame every N seconds; use 0 to disable.")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be >= 1")
    if args.snapshot_every < 0:
        parser.error("--snapshot-every must be >= 0")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(str(args.model))
    print(f"Input shape: {model.input_shape}; output shape: {model.output_shape}")
    if len(model.output_shape) < 2 or model.output_shape[-1] != 1:
        raise ValueError("Expected the project's single-output sigmoid CNN.")

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {args.camera}. Close apps using the webcam and retry.")

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir / session_id
    snapshots = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.snapshot_every:
        snapshots.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    rows: list[dict] = []
    history: deque[float] = deque(maxlen=3)
    start = time.monotonic()
    next_snapshot = start
    frame_index = 0

    print(f"Ground-truth session label: {args.label}")
    print(f"Capturing for {args.seconds}s. Press q to stop early.")
    print("Keep the scene/ground-truth label consistent for this entire session.")
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "frame", "elapsed_seconds", "label", "full_frame_fire_probability",
                "center_crop_fire_probability", "combined_fire_probability",
                "median3_fire_probability", "predicted_label_at_0.5",
                "correct_at_0.5", "mean_luminance",
            ])
            writer.writeheader()
            while time.monotonic() - start < args.seconds:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Camera read failed; stopping capture.")
                    break
                now = time.monotonic()
                full_p = fire_probability(model, frame)
                roi_p = fire_probability(model, center_crop(frame))
                combined_p = 0.65 * full_p + 0.35 * roi_p
                history.append(float(combined_p))
                median_p = float(statistics.median(history))
                predicted = "fire" if median_p >= 0.5 else "no_fire"
                elapsed = now - start
                correct = predicted == args.label
                luminance = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                row = {
                    "frame": frame_index,
                    "elapsed_seconds": round(elapsed, 3),
                    "label": args.label,
                    "full_frame_fire_probability": round(full_p, 6),
                    "center_crop_fire_probability": round(roi_p, 6),
                    "combined_fire_probability": round(combined_p, 6),
                    "median3_fire_probability": round(median_p, 6),
                    "predicted_label_at_0.5": predicted,
                    "correct_at_0.5": correct,
                    "mean_luminance": round(luminance, 3),
                }
                writer.writerow(row)
                file.flush()
                rows.append(row)

                display = frame.copy()
                color = (0, 180, 0) if predicted == args.label else (0, 0, 255)
                cv2.putText(display, f"GT: {args.label} | CNN: {median_p:.1%} | {predicted}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display, f"Full {full_p:.1%} | ROI {roi_p:.1%}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.imshow("CNN Webcam Diagnostic (q to stop)", display)

                if args.snapshot_every and now >= next_snapshot:
                    cv2.imwrite(str(snapshots / f"frame_{frame_index:05d}.jpg"), frame)
                    next_snapshot = now + args.snapshot_every
                frame_index += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames captured; no metrics generated.")
    probs = [float(r["median3_fire_probability"]) for r in rows]
    correct_count = sum(bool(r["correct_at_0.5"]) for r in rows)
    fire_predictions = sum(r["predicted_label_at_0.5"] == "fire" for r in rows)
    report = {
        "session_id_utc": session_id,
        "model_path": str(args.model),
        "ground_truth_label": args.label,
        "frames_evaluated": len(rows),
        "duration_seconds_requested": args.seconds,
        "decision_threshold": 0.5,
        "correct_frames_at_threshold": correct_count,
        "frame_level_accuracy": correct_count / len(rows),
        "fire_prediction_fraction": fire_predictions / len(rows),
        "median_fire_probability": statistics.median(probs),
        "mean_fire_probability": statistics.mean(probs),
        "p95_fire_probability": float(np.percentile(probs, 95)),
        "limitations": [
            "Frames from one continuous scene are correlated, not independent samples.",
            "This is a scene-specific diagnostic, not a certified real-world accuracy estimate.",
            "The output assumes class mapping fire=0 and no_fire=1 as used by this project.",
            "Do not compare this frame-level score directly with the existing offline image validation score.",
        ],
        "csv_path": str(csv_path),
        "snapshot_directory": str(snapshots) if args.snapshot_every else None,
    }
    report_path = out_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== CNN WEBCAM AUDIT SUMMARY ===")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
