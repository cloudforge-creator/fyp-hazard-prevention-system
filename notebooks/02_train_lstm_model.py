"""
02_train_lstm_model.py
Train the LSTM temperature anomaly predictor.

The runtime model (src/lstm_model.py) expects:
  models/lstm_temp_model.h5
  models/lstm_scaler.pkl
  models/lstm_threshold.pkl

Dataset:
  data/temperature_dataset/machine_temperature_system_failure.csv

The model learns to predict the next temperature from the previous
50 readings.  The anomaly threshold is calculated from validation
prediction error, so runtime inference uses the same scaled-error
space as training.
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 42
WINDOW_SIZE = 50
TEST_RATIO = 0.20
EPOCHS = 20
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    ROOT, "data", "temperature_dataset",
    "machine_temperature_system_failure.csv"
)
MODEL_DIR = os.path.join(ROOT, "models")
PLOT_DIR = os.path.join(ROOT, "notebooks", "plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("  LSTM Temperature Anomaly Model Training")
print("=" * 60)
print(f"Dataset: {DATA_PATH}")

# ------------------------------------------------------------------
# 1. Load and validate the temperature series
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
required = {"timestamp", "value"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns {required}; found {set(df.columns)}"
    )

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

values = df["value"].to_numpy(dtype=np.float32)

if len(values) <= WINDOW_SIZE + 100:
    raise ValueError("Not enough temperature readings for LSTM training.")

print(f"Readings: {len(values)}")
print(f"Temperature range: {values.min():.2f} to {values.max():.2f}")

# ------------------------------------------------------------------
# 2. Time-based train/validation split
#    Do not randomly shuffle time-series data.
# ------------------------------------------------------------------
split_idx = int(len(values) * (1.0 - TEST_RATIO))
train_values = values[:split_idx]
val_values = values[split_idx - WINDOW_SIZE:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_values.reshape(-1, 1)).ravel()
val_scaled = scaler.transform(val_values.reshape(-1, 1)).ravel()

def make_sequences(series, window):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i-window:i])
        y.append(series[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y

X_train, y_train = make_sequences(train_scaled, WINDOW_SIZE)
X_val, y_val = make_sequences(val_scaled, WINDOW_SIZE)

print(f"Training sequences:   {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")

# ------------------------------------------------------------------
# 3. Build LSTM
# ------------------------------------------------------------------
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    Dropout(0.20),
    LSTM(16),
    Dropout(0.20),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ------------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "lstm_temp_model.h5")

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------------
# 5. Evaluate next-temperature prediction
# ------------------------------------------------------------------
pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
mae_scaled = mean_absolute_error(y_val, pred_scaled)
rmse_scaled = np.sqrt(mean_squared_error(y_val, pred_scaled))

pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
y_actual = scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()

mae_actual = mean_absolute_error(y_actual, pred_actual)
rmse_actual = np.sqrt(mean_squared_error(y_actual, pred_actual))

# Runtime compares scaled absolute error against this threshold.
abs_errors = np.abs(pred_scaled - y_val)
threshold = float(np.percentile(abs_errors, 95))

print("\nEvaluation")
print(f"Scaled MAE:   {mae_scaled:.4f}")
print(f"Scaled RMSE:  {rmse_scaled:.4f}")
print(f"Actual MAE:   {mae_actual:.4f}")
print(f"Actual RMSE:  {rmse_actual:.4f}")
print(f"Anomaly threshold (95th percentile scaled error): {threshold:.4f}")

# ------------------------------------------------------------------
# 6. Save scaler + threshold
# ------------------------------------------------------------------
joblib.dump(scaler, os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
joblib.dump(threshold, os.path.join(MODEL_DIR, "lstm_threshold.pkl"))

print("\nSaved:")
print(f"  {model_path}")
print(f"  {os.path.join(MODEL_DIR, 'lstm_scaler.pkl')}")
print(f"  {os.path.join(MODEL_DIR, 'lstm_threshold.pkl')}")

# ------------------------------------------------------------------
# 7. Save training/evaluation plots
# ------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "lstm_training_loss.png"),
    dpi=150
)
plt.close()

plt.figure(figsize=(10, 4))
sample_n = min(500, len(y_actual))
plt.plot(y_actual[:sample_n], label="Actual")
plt.plot(pred_actual[:sample_n], label="Predicted")
plt.xlabel("Validation reading")
plt.ylabel("Temperature")
plt.title("LSTM Actual vs Predicted Temperature")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(PLOT_DIR, "lstm_prediction.png"),
    dpi=150
)
plt.close()

print(f"  {os.path.join(PLOT_DIR, 'lstm_training_loss.png')}")
print(f"  {os.path.join(PLOT_DIR, 'lstm_prediction.png')}")
print("\nLSTM training complete.")
