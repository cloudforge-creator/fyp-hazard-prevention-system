"""Evaluate the deployed CNN on separately collected, labeled unseen photos.

Put genuinely new photos into:
  data/cnn_unseen_photos/fire/
  data/cnn_unseen_photos/no_fire/

The script blocks byte-identical images already listed in the split manifest.
It uses MobileNetV2 preprocessing and the live inference full-frame/center-ROI
blend. Static images do not use the live stream's temporal median filter.
Run from repository root:
  python notebooks/09_evaluate_cnn_unseen_photos.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["fire", "no_fire"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMG_SIZE = (224, 224)
ROI_SCALE = 0.70


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest(value: str | None) -> Path:
    if value:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    for path in (
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_corrected.csv",
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_rebuilt.csv",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(
        "A split manifest is required for overlap checks. Generate the corrected "
        "manifest or pass --manifest PATH."
    )


def preprocess_bgr(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(image, IMG_SIZE), cv2.COLOR_BGR2RGB)
    return tf.keras.applications.mobilenet_v2.preprocess_input(
        rgb.astype(np.float32)
    )[None, ...]


def model_fire_probability(model, image: np.ndarray) -> float:
    """Match live inference's 65% full-frame + 35% center ROI blend."""
    h, w = image.shape[:2]
    crop_w, crop_h = max(1, int(w * ROI_SCALE)), max(1, int(h * ROI_SCALE))
    x1, y1 = max(0, (w - crop_w) // 2), max(0, (h - crop_h) // 2)
    roi = image[y1:y1 + crop_h, x1:x1 + crop_w]
    full_output = float(model.predict(preprocess_bgr(image), verbose=0).reshape(-1)[0])
    roi_output = float(model.predict(preprocess_bgr(roi), verbose=0).reshape(-1)[0])
    # The sigmoid output is P(no_fire), because the trained class indices are fire=0, no_fire=1.
    full_fire = 1.0 - np.clip(full_output, 0.0, 1.0)
    roi_fire = 1.0 - np.clip(roi_output, 0.0, 1.0)
    return float(np.clip(0.65 * full_fire + 0.35 * roi_fire, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-dir", default="data/cnn_unseen_photos")
    parser.add_argument("--model", default="models/cnn_fire_model.h5")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", default="notebooks/plots/cnn_unseen_photos")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")

    photos_dir = Path(args.photos_dir)
    if not photos_dir.is_absolute():
        photos_dir = ROOT / photos_dir
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    manifest_path = resolve_manifest(args.manifest)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not photos_dir.is_dir():
        raise FileNotFoundError(
            f"Unseen-photo directory not found: {photos_dir}. Create fire/ and no_fire/ subfolders."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if "relative_path" not in manifest.columns:
        raise ValueError("Manifest must contain a relative_path column.")

    known_hashes: dict[str, str] = {}
    for relative in manifest["relative_path"].dropna().astype(str):
        known_path = (ROOT / relative).resolve()
        if known_path.is_file():
            known_hashes.setdefault(sha256_file(known_path), relative)

    photo_rows: list[tuple[Path, str]] = []
    for label in CLASS_NAMES:
        folder = photos_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        photo_rows.extend(
            (path, label) for path in sorted(folder.rglob("*"))
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
    if not photo_rows:
        raise ValueError(f"No labeled images found under {photos_dir}. Add photos to fire/ and no_fire/.")

    seen_new_hashes: set[str] = set()
    overlaps: list[str] = []
    for path, _ in photo_rows:
        digest = sha256_file(path)
        if digest in known_hashes:
            overlaps.append(f"{path} duplicates manifest image {known_hashes[digest]}")
        if digest in seen_new_hashes:
            overlaps.append(f"{path} duplicates another unseen-photo file")
        seen_new_hashes.add(digest)
    if overlaps:
        raise ValueError(
            "Unseen-photo evaluation blocked because exact duplicate files were found:\n"
            + "\n".join(overlaps[:20])
            + "\nRemove duplicates and rerun. Exact-byte checks do not detect resized/near-duplicate images."
        )

    model = tf.keras.models.load_model(str(model_path))
    rows = []
    y_true, y_pred = [], []
    for path, label in photo_rows:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
        fire_probability = model_fire_probability(model, image)
        truth = CLASS_NAMES.index(label)
        predicted = 0 if fire_probability >= args.threshold else 1
        predicted_label = CLASS_NAMES[predicted]
        y_true.append(truth)
        y_pred.append(predicted)
        rows.append({
            "image": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "true_label": label,
            "fire_probability": round(fire_probability, 6),
            "predicted_label": predicted_label,
            "correct": label == predicted_label,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    result = {
        "model_path": str(model_path),
        "photos_directory": str(photos_dir),
        "manifest_path": str(manifest_path),
        "evaluation": "separate labeled unseen photos; not the original 570-image held-out test",
        "inference": "MobileNetV2 preprocess_input; P(fire)=1-P(no_fire); 0.65 full-frame + 0.35 center ROI; no temporal filter for static photos",
        "sample_count": len(rows),
        "threshold": args.threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_indices": {"fire": 0, "no_fire": 1},
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "exact_duplicate_check": "passed against manifest-listed images and within unseen-photo set",
        "note": "Exact-byte duplicate checking cannot identify resized or visually near-duplicate photos; confirm photos were newly captured and manually review provenance.",
    }
    (output_dir / "unseen_photo_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_dir / "unseen_photo_predictions.csv", index=False)
    print(f"Unseen-photo accuracy: {result['accuracy']:.4%} ({len(rows)} images)")
    print(f"Fire recall: {report['fire']['recall']:.4%}")
    print(f"Reports saved in: {output_dir}")
    print(json.dumps(matrix.tolist(), indent=2))


if __name__ == "__main__":
    main()
