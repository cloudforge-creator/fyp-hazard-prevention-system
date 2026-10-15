"""Audit fire/no-fire image quality and evaluate CNN on a duplicate-grouped holdout.

Run from repository root:
    python notebooks/05_cnn_quality_audit.py
    python notebooks/05_cnn_quality_audit.py --phash-distance 5

This script does not train or replace a model. The holdout is created from the
local dataset and is not an external real-world test set. It uses MobileNetV2
preprocessing and the project's class order: fire=0, no_fire=1 (sigmoid output
is interpreted as no_fire probability).
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
DEFAULT_MODEL = ROOT / "models" / "cnn_fire_model.h5"
DEFAULT_OUTPUT = ROOT / "notebooks" / "plots" / "cnn_quality_audit.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR):
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, mode)
    except Exception:
        return None


def phash(image: np.ndarray) -> int:
    """Simple 64-bit perceptual hash using low-frequency DCT coefficients."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))  # exclude the DC coefficient
    bits = low > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def audit_images(dataset_dir: Path, max_distance: int):
    classes = ["fire", "no_fire"]
    records = []
    unreadable = []
    class_file_counts = {}

    for class_name in classes:
        folder = dataset_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Required class folder is missing: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        class_file_counts[class_name] = len(paths)
        for path in paths:
            image = read_image(path)
            if image is None:
                unreadable.append(str(path.relative_to(ROOT)))
                continue
            height, width = image.shape[:2]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append({
                "path": path,
                "relative_path": str(path.relative_to(ROOT)),
                "label": 0 if class_name == "fire" else 1,
                "class_name": class_name,
                "sha256": digest,
                "phash": phash(image),
                "width": int(width),
                "height": int(height),
                "bytes": path.stat().st_size,
            })

    if len(records) < 20:
        raise RuntimeError(f"Only {len(records)} readable images found; cannot audit reliably.")

    exact_groups = {}
    for i, item in enumerate(records):
        exact_groups.setdefault(item["sha256"], []).append(i)
    exact_duplicate_groups = [v for v in exact_groups.values() if len(v) > 1]
    exact_duplicate_files = sum(len(v) - 1 for v in exact_duplicate_groups)

    # Union-find groups visually similar images so a duplicate family cannot
    # appear in both the training and holdout partitions.
    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = [r["phash"] for r in records]
    near_duplicate_pairs = 0
    cross_class_near_pairs = []
    print(f"Comparing perceptual hashes for {len(records):,} readable images...")
    for i in range(len(records)):
        hi = hashes[i]
        li = records[i]["label"]
        for j in range(i + 1, len(records)):
            distance = (hi ^ hashes[j]).bit_count()
            if distance <= max_distance:
                near_duplicate_pairs += 1
                union(i, j)
                if li != records[j]["label"]:
                    cross_class_near_pairs.append({
                        "distance": distance,
                        "image_a": records[i]["relative_path"],
                        "class_a": records[i]["class_name"],
                        "image_b": records[j]["relative_path"],
                        "class_b": records[j]["class_name"],
                    })

    groups = [find(i) for i in range(len(records))]
    unique_groups = sorted(set(groups))
    group_to_id = {root: idx for idx, root in enumerate(unique_groups)}
    group_ids = np.asarray([group_to_id[root] for root in groups], dtype=np.int64)
    labels = np.asarray([r["label"] for r in records], dtype=np.int32)

    # Dimensions and file-size summaries help identify inconsistent source data.
    dimensions = Counter(f"{r['width']}x{r['height']}" for r in records)
    class_valid_counts = Counter(r["class_name"] for r in records)

    audit = {
        "dataset_directory": str(dataset_dir),
        "model_path": str(DEFAULT_MODEL),
        "class_mapping": {"fire": 0, "no_fire": 1},
        "raw_file_counts": class_file_counts,
        "readable_image_counts": dict(class_valid_counts),
        "readable_total": len(records),
        "unreadable_count": len(unreadable),
        "unreadable_examples": unreadable[:30],
        "exact_duplicate_groups": len(exact_duplicate_groups),
        "exact_duplicate_extra_files": exact_duplicate_files,
        "perceptual_hash_distance_threshold": max_distance,
        "near_duplicate_pairs": near_duplicate_pairs,
        "near_duplicate_group_count": len(unique_groups),
        "cross_class_near_duplicate_pair_count": len(cross_class_near_pairs),
        "cross_class_near_duplicate_examples": cross_class_near_pairs[:50],
        "image_dimensions_top_20": dimensions.most_common(20),
        "grouped_holdout": None,
        "limitations": [
            "Perceptual-hash matches are candidates for review, not proof of duplicates.",
            "The grouped holdout is drawn from this dataset and is not an external real-world test.",
            "CNN output is interpreted using the project's expected sigmoid convention: output=probability of no_fire.",
        ],
    }
    return records, labels, group_ids, audit


def evaluate_grouped_holdout(records, labels, group_ids, model_path: Path, seed: int):
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import tensorflow as tf

    if not model_path.is_file():
        raise FileNotFoundError(f"CNN model not found: {model_path}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels, groups=group_ids))

    if len(set(labels[test_idx])) < 2:
        raise RuntimeError("Grouped holdout contains only one class; inspect duplicates or use another seed.")
    overlap = set(group_ids[train_idx]).intersection(set(group_ids[test_idx]))
    if overlap:
        raise AssertionError("Duplicate-group leakage detected between train and holdout.")

    model = tf.keras.models.load_model(str(model_path), compile=False)
    shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    if len(shape) != 4 or shape[-1] != 3:
        raise ValueError(f"Expected a 3-channel image model, got input shape {shape}")
    height, width = int(shape[1]), int(shape[2])

    images = []
    kept_indices = []
    for idx in test_idx:
        image = read_image(records[idx]["path"])
        if image is None:
            continue
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image.astype(np.float32))
        kept_indices.append(idx)

    if not images:
        raise RuntimeError("No holdout images could be decoded for model evaluation.")

    batch = tf.keras.applications.mobilenet_v2.preprocess_input(np.asarray(images))
    raw_no_fire = np.asarray(model.predict(batch, verbose=0)).reshape(-1)
    if len(raw_no_fire) != len(kept_indices):
        raise RuntimeError("Model returned an unexpected number of predictions.")
    if np.any(~np.isfinite(raw_no_fire)):
        raise RuntimeError("Model returned NaN or infinite predictions.")

    truth = labels[kept_indices]
    # Training uses alphabetical directory labels: fire=0 and no_fire=1.
    predicted = (raw_no_fire >= 0.5).astype(np.int32)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    fire_total = int(matrix[0].sum())
    no_fire_total = int(matrix[1].sum())
    false_fire = int(matrix[1, 0])
    return {
        "model_input_shape": list(shape),
        "holdout_seed": seed,
        "holdout_image_count": int(len(truth)),
        "holdout_group_count": int(len(set(group_ids[kept_indices]))),
        "train_group_count": int(len(set(group_ids[train_idx]))),
        "group_overlap_count": len(overlap),
        "holdout_class_counts": {
            "fire": int(np.sum(truth == 0)),
            "no_fire": int(np.sum(truth == 1)),
        },
        "accuracy": float(accuracy_score(truth, predicted)),
        "fire_recall": float(matrix[0, 0] / fire_total) if fire_total else None,
        "no_fire_recall_specificity": float(matrix[1, 1] / no_fire_total) if no_fire_total else None,
        "false_fire_count": false_fire,
        "false_fire_rate_on_no_fire": float(false_fire / no_fire_total) if no_fire_total else None,
        "confusion_matrix_labels_order": ["fire", "no_fire"],
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            truth, predicted, labels=[0, 1], target_names=["fire", "no_fire"],
            output_dict=True, zero_division=0,
        ),
        "threshold": 0.5,
        "interpretation_note": "Grouped holdout is more resistant to duplicate leakage, but is not external validation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phash-distance", type=int, default=5,
                        help="Maximum 64-bit perceptual-hash Hamming distance for candidate duplicates (default: 5).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.phash_distance < 0 or args.phash_distance > 16:
        parser.error("--phash-distance must be between 0 and 16")

    records, labels, group_ids, audit = audit_images(args.dataset, args.phash_distance)
    audit["model_path"] = str(args.model)
    audit["grouped_holdout"] = evaluate_grouped_holdout(
        records, labels, group_ids, args.model, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n=== CNN DATASET QUALITY AUDIT ===")
    print("Raw class counts:", audit["raw_file_counts"])
    print("Readable class counts:", audit["readable_image_counts"])
    print("Unreadable images:", audit["unreadable_count"])
    print("Exact duplicate extra files:", audit["exact_duplicate_extra_files"])
    print("Near-duplicate pairs:", audit["near_duplicate_pairs"])
    print("Cross-class near-duplicate pairs:", audit["cross_class_near_duplicate_pair_count"])
    result = audit["grouped_holdout"]
    print("\n=== DUPLICATE-GROUPED HOLDOUT (NOT EXTERNAL TEST) ===")
    print(f"Accuracy: {result['accuracy']:.4%}")
    print(f"Fire recall: {result['fire_recall']:.4%}")
    print(f"No-fire recall/specificity: {result['no_fire_recall_specificity']:.4%}")
    print(f"False fire: {result['false_fire_count']}/{result['holdout_class_counts']['no_fire']} "
          f"({result['false_fire_rate_on_no_fire']:.2%})")
    print("Confusion matrix [fire, no_fire]:", result["confusion_matrix"])
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
