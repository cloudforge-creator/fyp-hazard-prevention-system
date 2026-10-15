"""Audit fire/no-fire dataset counts, image integrity, duplicates, and split leakage.

Run from repository root:
    python notebooks/05_cnn_quality_audit.py
    python notebooks/05_cnn_quality_audit.py --phash-distance 5

This is a data audit only: it does not train, modify, or evaluate a model.
It reconstructs the training script's deterministic per-class 80/20 filename
split to identify exact/perceptual duplicate candidates that cross that split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "fire_dataset"
DEFAULT_OUTPUT = ROOT / "notebooks" / "plots" / "cnn_quality_audit.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
TRAIN_FRACTION = 0.8


def read_image(path: Path):
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def phash(image: np.ndarray) -> int:
    """Simple 64-bit perceptual hash using low-frequency DCT coefficients."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))  # exclude DC coefficient
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bit)
    return value


def audit_images(dataset_dir: Path, max_distance: int) -> dict:
    records = []
    unreadable = []
    raw_counts = {}
    valid_counts = {}
    dimensions = Counter()

    # Match the project's alphabetical class mapping and per-class filename split.
    for class_name, label in (("fire", 0), ("no_fire", 1)):
        folder = dataset_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required class folder is missing: {folder}")
        paths = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        raw_counts[class_name] = len(paths)
        split_boundary = int(len(paths) * TRAIN_FRACTION)
        valid_counts[class_name] = 0

        for index, path in enumerate(paths):
            image = read_image(path)
            if image is None:
                unreadable.append(display_path(path))
                continue
            height, width = image.shape[:2]
            valid_counts[class_name] += 1
            dimensions[f"{width}x{height}"] += 1
            records.append({
                "path": path,
                "relative_path": display_path(path),
                "class_name": class_name,
                "label": label,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "phash": phash(image),
                "width": int(width),
                "height": int(height),
                "split": "train" if index < split_boundary else "validation",
            })

    if not records:
        raise RuntimeError("No readable images were found.")

    exact_groups = {}
    for i, item in enumerate(records):
        exact_groups.setdefault(item["sha256"], []).append(i)
    exact_duplicate_groups = [items for items in exact_groups.values() if len(items) > 1]
    exact_duplicate_extra_files = sum(len(items) - 1 for items in exact_duplicate_groups)
    exact_cross_split_pairs = []
    for items in exact_duplicate_groups:
        for pos, i in enumerate(items):
            for j in items[pos + 1:]:
                if records[i]["split"] != records[j]["split"]:
                    exact_cross_split_pairs.append({
                        "image_a": records[i]["relative_path"],
                        "image_b": records[j]["relative_path"],
                        "class_a": records[i]["class_name"],
                        "class_b": records[j]["class_name"],
                        "split_a": records[i]["split"],
                        "split_b": records[j]["split"],
                    })

    # Pairwise perceptual-hash comparison. Similarity is a review signal, not
    # proof of duplicate content or a labeling error.
    hashes = [r["phash"] for r in records]
    near_pairs = 0
    near_cross_split = []
    near_cross_class = []
    print(f"Comparing perceptual hashes for {len(records):,} readable images...")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            distance = (hashes[i] ^ hashes[j]).bit_count()
            if distance > max_distance:
                continue
            a, b = records[i], records[j]
            near_pairs += 1
            pair = {
                "distance": distance,
                "image_a": a["relative_path"],
                "image_b": b["relative_path"],
                "class_a": a["class_name"],
                "class_b": b["class_name"],
                "split_a": a["split"],
                "split_b": b["split"],
            }
            if a["split"] != b["split"]:
                near_cross_split.append(pair)
            if a["label"] != b["label"]:
                near_cross_class.append(pair)

    exact_cross_class_groups = [
        items for items in exact_duplicate_groups
        if len({records[i]["label"] for i in items}) > 1
    ]
    readable_total = len(records)
    count_min = min(valid_counts.values()) if valid_counts else 0

    return {
        "dataset_directory": str(dataset_dir),
        "class_mapping": {"fire": 0, "no_fire": 1},
        "raw_file_counts": raw_counts,
        "readable_image_counts": valid_counts,
        "readable_total": readable_total,
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:50],
        "class_count_difference": abs(valid_counts.get("fire", 0) - valid_counts.get("no_fire", 0)),
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "exact_duplicate_extra_files": exact_duplicate_extra_files,
        "exact_duplicate_pairs_crossing_train_validation": len(exact_cross_split_pairs),
        "exact_duplicate_cross_split_examples": exact_cross_split_pairs[:50],
        "exact_duplicate_groups_with_cross_class_labels": len(exact_cross_class_groups),
        "phash_distance_threshold": max_distance,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_pairs_crossing_train_validation": len(near_cross_split),
        "near_duplicate_cross_split_examples": near_cross_split[:50],
        "near_duplicate_pairs_with_cross_class_labels": len(near_cross_class),
        "near_duplicate_cross_class_examples": near_cross_class[:50],
        "image_dimensions_top_20": dimensions.most_common(20),
        "training_split_reconstruction": (
            "Sorted filenames within each class; first 80% train and final 20% validation. "
            "This mirrors the project's flow_from_directory(validation_split=0.2) intent; "
            "confirm against the installed Keras version if exact membership is critical."
        ),
        "dataset_size_note": (
            "Image count alone cannot establish dataset sufficiency. Diversity, labels, "
            "source independence, duplicate leakage, and performance on unseen webcam scenes matter."
        ),
        "limitations": [
            "Perceptual-hash matches are candidates for manual review, not proof of duplicates.",
            "This script audits the dataset; it does not prove model generalization or certify safety.",
            "A separate real-world test set of unseen images and labeled webcam clips is still required.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phash-distance", type=int, default=5,
        help="Maximum 64-bit perceptual-hash Hamming distance for candidate duplicates (default: 5).",
    )
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 16:
        parser.error("--phash-distance must be between 0 and 16")

    report = audit_images(args.dataset, args.phash_distance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== CNN DATASET QUALITY AUDIT ===")
    print("Raw image counts:", report["raw_file_counts"])
    print("Readable image counts:", report["readable_image_counts"])
    print("Unreadable images:", report["unreadable_count"])
    print("Exact duplicate extra files:", report["exact_duplicate_extra_files"])
    print("Exact duplicate pairs crossing train/validation:",
          report["exact_duplicate_pairs_crossing_train_validation"])
    print("Perceptual near-duplicate pairs:", report["near_duplicate_pairs"])
    print("Near-duplicate pairs crossing train/validation:",
          report["near_duplicate_pairs_crossing_train_validation"])
    print("Near-duplicate pairs across fire/no-fire labels:",
          report["near_duplicate_pairs_with_cross_class_labels"])
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
