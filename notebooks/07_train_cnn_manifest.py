"""Train a candidate MobileNetV2 CNN using an explicit group-safe manifest.

The test split is never passed to model.fit() and is not evaluated here.
Run from the repository root:
    python notebooks/07_train_cnn_manifest.py
Optional:
    python notebooks/07_train_cnn_manifest.py --manifest notebooks/plots/cnn_split_manifest_rebuilt.csv
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
CLASS_NAMES = ["fire", "no_fire"]


def resolve_manifest(value: str | None) -> Path:
    if value:
        return Path(value)
    candidates = [
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_corrected.csv",
        ROOT / "notebooks" / "plots" / "cnn_split_manifest_rebuilt.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No CNN split manifest found. Run "
        "'python notebooks/06_build_cnn_split_manifest.py' first, or pass --manifest."
    )


def load_and_validate_manifest(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path)
    required = {"relative_path", "label", "group_id", "split"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
    if manifest.empty:
        raise ValueError("Manifest is empty.")
    if manifest["relative_path"].isna().any() or manifest["relative_path"].duplicated().any():
        raise ValueError("Manifest paths contain blanks or duplicates.")
    if not set(manifest["label"].astype(str)).issubset(CLASS_NAMES):
        raise ValueError("Manifest labels must be 'fire' or 'no_fire'.")
    if not set(manifest["split"].astype(str)).issubset({"train", "validation", "test"}):
        raise ValueError("Manifest contains an unknown split.")
    if manifest[["group_id", "split"]].isna().any().any():
        raise ValueError("Manifest group IDs and split names cannot be blank.")

    group_split_counts = manifest.groupby("group_id")["split"].nunique()
    if (group_split_counts > 1).any():
        bad_groups = group_split_counts[group_split_counts > 1].index.tolist()[:10]
        raise ValueError(f"Duplicate groups cross splits: {bad_groups}")

    # Resolve paths relative to repository root, not the caller's working directory.
    resolved = []
    missing = []
    for item in manifest["relative_path"].astype(str):
        p = (ROOT / item).resolve()
        resolved.append(str(p))
        if not p.is_file():
            missing.append(item)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest image files are missing; examples: {missing[:5]}"
        )
    manifest = manifest.copy()
    manifest["absolute_path"] = resolved

    splits = {
        name: manifest.loc[manifest["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    if any(frame.empty for frame in splits.values()):
        raise ValueError("Train, validation, and test splits must all be non-empty.")
    for name, frame in splits.items():
        missing_classes = set(CLASS_NAMES) - set(frame["label"].astype(str))
        if missing_classes:
            raise ValueError(f"{name} split is missing classes: {sorted(missing_classes)}")

    print(f"Manifest: {path}")
    print("\nCounts by split and class:")
    print(pd.crosstab(manifest["split"], manifest["label"]))
    print(f"Total records: {len(manifest)}")
    return manifest, splits


def make_generator(frame: pd.DataFrame, augment: bool):
    if augment:
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.7, 1.3],
        )
    else:
        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        )
    return generator.flow_from_dataframe(
        dataframe=frame,
        directory=None,
        x_col="absolute_path",
        y_col="label",
        classes=CLASS_NAMES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=augment,
        seed=SEED if augment else None,
        validate_filenames=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="CSV split manifest. Auto-detects corrected/rebuilt manifest if omitted.")
    parser.add_argument("--epochs-head", type=int, default=10)
    parser.add_argument("--epochs-finetune", type=int, default=10)
    parser.add_argument("--output-model", default="models/cnn_fire_model_candidate.h5")
    parser.add_argument("--checkpoint", default="models/cnn_fire_model_candidate_best.h5")
    parser.add_argument("--output-dir", default="notebooks/plots/cnn_manifest_run")
    args = parser.parse_args()
    if args.epochs_head < 1 or args.epochs_finetune < 1:
        parser.error("Epoch counts must be positive integers.")

    manifest_path = resolve_manifest(args.manifest)
    manifest, splits = load_and_validate_manifest(manifest_path)
    train_data = make_generator(splits["train"], augment=True)
    val_data = make_generator(splits["validation"], augment=False)
    print(f"Class mapping: {train_data.class_indices}")
    if train_data.class_indices != {"fire": 0, "no_fire": 1}:
        raise ValueError(f"Unexpected class mapping: {train_data.class_indices}")
    print(f"Test images held aside (not used in training): {len(splits['test'])}")

    classes = np.unique(train_data.classes)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_data.classes
    )
    class_weights = {int(k): float(v) for k, v in zip(classes, weights)}
    print(f"Class weights: {class_weights}")

    tf.keras.utils.set_random_seed(SEED)
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    output_model = ROOT / args.output_model
    checkpoint = ROOT / args.checkpoint
    output_dir = ROOT / args.output_dir
    output_model.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_model.exists() or checkpoint.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing candidate model/checkpoint. "
            "Choose new --output-model and --checkpoint paths."
        )

    best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint), monitor="val_accuracy", save_best_only=True,
        mode="max", verbose=1,
    )
    history_head = model.fit(
        train_data,
        epochs=args.epochs_head,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3,
                restore_best_weights=True, verbose=1,
            ),
            best_checkpoint,
        ],
    )

    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history_finetune = model.fit(
        train_data,
        epochs=args.epochs_finetune,
        validation_data=val_data,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=5,
                restore_best_weights=True, verbose=1,
            ),
            best_checkpoint,
        ],
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training produced no best checkpoint: {checkpoint}")
    model = tf.keras.models.load_model(str(checkpoint))
    model.save(str(output_model))

    val_data.reset()
    val_loss, val_accuracy = model.evaluate(val_data, verbose=0)
    val_data.reset()
    probabilities = model.predict(val_data, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    truth = val_data.classes
    report = classification_report(
        truth, predictions, labels=[0, 1], target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN Candidate - Validation Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / "validation_confusion_matrix.png", dpi=150)
    plt.close()

    train_acc = history_head.history.get("accuracy", []) + history_finetune.history.get("accuracy", [])
    val_acc = history_head.history.get("val_accuracy", []) + history_finetune.history.get("val_accuracy", [])
    plt.figure(figsize=(10, 4))
    plt.plot(train_acc, label="Train accuracy")
    plt.plot(val_acc, label="Validation accuracy")
    plt.axvline(max(len(history_head.history.get("accuracy", [])) - 1, 0),
                linestyle="--", label="Fine-tuning starts")
    plt.title("CNN Candidate Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()

    result = {
        "model": "MobileNetV2 CNN candidate",
        "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "manifest_rows": int(len(manifest)),
        "split_counts": {name: int(len(frame)) for name, frame in splits.items()},
        "validation_sample_count": int(len(truth)),
        "validation_loss": float(val_loss),
        "validation_accuracy": float(val_accuracy),
        "threshold": 0.5,
        "class_indices": train_data.class_indices,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "candidate_model": str(output_model.relative_to(ROOT)),
        "best_checkpoint": str(checkpoint.relative_to(ROOT)),
        "test_split_used_during_training": False,
        "note": "Validation metrics select the candidate. Evaluate the untouched test split only after model selection.",
    }
    (output_dir / "training_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps({
        "head": history_head.history,
        "fine_tune": history_finetune.history,
    }, indent=2), encoding="utf-8")
    print(f"\nCandidate saved: {output_model}")
    print(f"Validation accuracy: {val_accuracy:.4%} ({len(truth)} images)")
    print(f"Test set was not evaluated: {len(splits['test'])} images remain held out.")
    print(f"Metrics and plots: {output_dir}")


if __name__ == "__main__":
    main()
