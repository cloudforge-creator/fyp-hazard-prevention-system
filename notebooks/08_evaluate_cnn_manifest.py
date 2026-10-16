"""Evaluate a selected CNN model against an explicit manifest split.

Default is the untouched test split. Do not tune the model or threshold using
these results; run this once after selecting a candidate on validation.
Run from repository root:
    python notebooks/08_evaluate_cnn_manifest.py --model models/cnn_fire_model_candidate.h5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["fire", "no_fire"]


def find_manifest(value: str | None) -> Path:
    if value:
        return Path(value)
    for path in (
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_corrected.csv",
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_rebuilt.csv",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No split manifest found. Run notebooks/06_build_cnn_split_manifest.py first."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="CSV manifest; auto-detected if omitted.")
    parser.add_argument("--model", default="models/cnn_fire_model_candidate.h5")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fixed probability threshold. Do not tune on the test split.")
    parser.add_argument("--output-dir", default="notebooks/plots/cnn_manifest_test")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")

    manifest_path = find_manifest(args.manifest)
    model_path = ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    required = {"relative_path", "label", "group_id", "split"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if manifest["relative_path"].isna().any() or manifest["relative_path"].duplicated().any():
        raise ValueError("Manifest paths contain blanks or duplicates.")
    if not set(manifest["label"].astype(str)).issubset(CLASS_NAMES):
        raise ValueError("Manifest contains unexpected labels.")
    if (manifest.groupby("group_id")["split"].nunique() > 1).any():
        raise ValueError("Manifest has duplicate groups crossing splits.")

    selected = manifest.loc[manifest["split"] == args.split].copy()
    if selected.empty:
        raise ValueError(f"Manifest has no rows for split '{args.split}'.")
    missing = [
        item for item in selected["relative_path"].astype(str)
        if not (ROOT / item).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files from split '{args.split}' are missing: {missing[:5]}"
        )
    if set(selected["label"].astype(str)) != set(CLASS_NAMES):
        raise ValueError(f"Split '{args.split}' must contain both classes.")

    selected["absolute_path"] = [
        str((ROOT / item).resolve()) for item in selected["relative_path"].astype(str)
    ]
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )
    data = generator.flow_from_dataframe(
        dataframe=selected,
        directory=None,
        x_col="absolute_path",
        y_col="label",
        classes=CLASS_NAMES,
        target_size=(224, 224),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
        validate_filenames=True,
    )
    model = tf.keras.models.load_model(str(model_path))
    probabilities = model.predict(data, verbose=0).reshape(-1)
    predictions = (probabilities >= args.threshold).astype(int)
    truth = data.classes
    report = classification_report(
        truth, predictions, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"CNN Candidate - {args.split.title()} Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.split}_confusion_matrix.png", dpi=150)
    plt.close()

    result = {
        "model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "split": args.split,
        "sample_count": int(len(truth)),
        "class_indices": data.class_indices,
        "threshold": float(args.threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "note": (
            "Independent test metrics; do not use for threshold/epoch/model tuning."
            if args.split == "test"
            else "Validation metrics for model selection; not independent test proof."
        ),
    }
    result_path = output_dir / f"{args.split}_evaluation.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{args.split.title()} accuracy: {result['accuracy']:.4%} ({len(truth)} images)")
    print(json.dumps(report, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
