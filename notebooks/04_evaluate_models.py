"""Evaluate the trained CNN and optionally supplied labeled gas/temperature datasets.

CNN results use the same deterministic 20% validation split as training; they are
not independent test metrics. Use an untouched test set for final claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def evaluate_cnn(dataset_dir: Path, model_path: Path, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    if not model_path.is_file():
        raise FileNotFoundError(f"CNN model not found: {model_path}")
    if not (dataset_dir / "fire").is_dir() or not (dataset_dir / "no_fire").is_dir():
        raise FileNotFoundError(f"Expected fire and no_fire folders under {dataset_dir}")
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0, validation_split=0.2
    )
    data = generator.flow_from_directory(
        str(dataset_dir), target_size=(224, 224), batch_size=32,
        class_mode="binary", subset="validation", seed=42, shuffle=False
    )
    model = tf.keras.models.load_model(str(model_path))
    data.reset()
    probabilities = model.predict(data, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    truth = data.classes
    names = [name for name, _ in sorted(data.class_indices.items(), key=lambda item: item[1])]
    report = classification_report(truth, predictions, target_names=names,
                                    output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=list(range(len(names))))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.title("CNN Fire Detection - Validation Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_dir / "cnn_confusion_matrix.png", dpi=160)
    plt.close()
    result = {
        "model": "MobileNetV2 CNN",
        "split": "deterministic 20% validation split; not an independent test set",
        "dataset_directory": str(dataset_dir), "model_path": str(model_path),
        "sample_count": int(len(truth)), "class_indices": data.class_indices,
        "accuracy": float(accuracy_score(truth, predictions)),
        "classification_report": report, "confusion_matrix": matrix.tolist(),
        "threshold": 0.5,
    }
    (output_dir / "cnn_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CNN validation accuracy: {result['accuracy']:.4%} ({len(truth)} images)")
    print(json.dumps(report, indent=2))
    print(f"Saved confusion matrix and metrics in {output_dir}")
    return result


def evaluate_gas(csv_path: Path, output_dir: Path) -> dict:
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from src.gas_model import GasModel
    df = pd.read_csv(csv_path)
    if not {"gas_ratio", "gas_raw", "label"}.issubset(df.columns):
        raise ValueError("Gas CSV must contain gas_ratio,gas_raw,label columns")
    model = GasModel()
    if not model.model_loaded:
        raise RuntimeError("Trained gas model did not load; refusing to report ML accuracy")
    truth = df["label"].astype(str).tolist()
    predicted = [model.predict(float(r.gas_ratio), int(r.gas_raw))[0] for r in df.itertuples()]
    labels = sorted(set(truth) | set(predicted))
    result = {
        "model": "Random Forest gas classifier", "csv": str(csv_path),
        "sample_count": len(df), "accuracy": float(accuracy_score(truth, predicted)),
        "labels": labels,
        "classification_report": classification_report(truth, predicted, labels=labels,
                                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gas_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Gas model accuracy: {result['accuracy']:.4%} ({len(df)} labeled rows)")
    return result


def evaluate_lstm(csv_path: Path, output_dir: Path) -> dict:
    import pandas as pd
    import joblib
    import tensorflow as tf
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    model_path, scaler_path = Path("models/lstm_temp_model.h5"), Path("models/lstm_scaler.pkl")
    if not model_path.is_file() or not scaler_path.is_file():
        raise FileNotFoundError("Trained LSTM model and scaler are required")
    df = pd.read_csv(csv_path)
    if not {"timestamp", "value"}.issubset(df.columns):
        raise ValueError("Temperature CSV must contain timestamp,value columns")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    values = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")["value"].to_numpy(dtype=np.float32)
    window, split = 50, int(len(values) * 0.8)
    if split <= window or len(values) <= split:
        raise ValueError("Not enough valid readings for chronological 80/20 evaluation")
    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(str(model_path))
    scaled = scaler.transform(values.reshape(-1, 1)).ravel()
    X = np.asarray([scaled[i-window:i] for i in range(split, len(scaled))],
                   dtype=np.float32).reshape(-1, window, 1)
    truth = scaler.inverse_transform(scaled[split:].reshape(-1, 1)).ravel()
    pred_scaled = model.predict(X, verbose=0).reshape(-1)
    predicted = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    result = {
        "model": "LSTM temperature predictor", "csv": str(csv_path),
        "split": "chronological final 20%", "sample_count": int(len(truth)),
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "unit": "same as source CSV",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lstm_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"LSTM holdout MAE: {result['mae']:.4f}; RMSE: {result['rmse']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-dataset", default="data/fire_dataset")
    parser.add_argument("--cnn-model", default="models/cnn_fire_model.h5")
    parser.add_argument("--gas-csv", help="Optional labeled CSV: gas_ratio,gas_raw,label")
    parser.add_argument("--temperature-csv", help="Optional CSV with timestamp,value")
    parser.add_argument("--output-dir", default="notebooks/plots")
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    results = []
    if not args.skip_cnn:
        results.append(evaluate_cnn(Path(args.cnn_dataset), Path(args.cnn_model), output))
    if args.gas_csv:
        results.append(evaluate_gas(Path(args.gas_csv), output))
    if args.temperature_csv:
        results.append(evaluate_lstm(Path(args.temperature_csv), output))
    summary = {
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "note": "Metric validity depends on labeled representative data; validation is not independent test proof.",
        "evaluations": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluation summary: {output / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
