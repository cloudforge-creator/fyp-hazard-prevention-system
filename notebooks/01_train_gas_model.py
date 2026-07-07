"""
01_train_gas_model.py
Train the Random Forest gas classifier for MQ-2 sensor.

IMPORTANT: Because MQ-2 gives one analogue reading (not 128 channels
like the UCI dataset), we train on a synthetic dataset built from
MQ-2 datasheet characteristics + your sensor's calibration readings.

When you have hardware:
  1. Run the Arduino sketch
  2. Note the GAS_RATIO values in Serial Monitor for clean air
  3. Update CLEAN_AIR_RATIO below
  4. Run this script to retrain

Run: python notebooks/01_train_gas_model.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,
                              confusion_matrix, accuracy_score)
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks/plots', exist_ok=True)

# ---- MQ-2 Datasheet Rs/Ro Ratios ----
# Source: MQ-2 datasheet sensitivity curves
# Format: (gas_label, ratio_mean, ratio_std, n_samples, risk_level)
MQ2_GAS_PROFILES = [
    ('clean_air',  10.0, 2.0,  800, 0),
    ('smoke',       5.0, 1.2,  400, 1),
    ('co',          2.5, 0.6,  350, 2),
    ('lpg',         1.5, 0.35, 400, 3),
    ('methane',     1.0, 0.25, 350, 4),
    ('hydrogen',    0.5, 0.15, 300, 5),
]

GAS_NAMES  = [p[0] for p in MQ2_GAS_PROFILES]
GAS_LABELS = {name: i for i, name in enumerate(GAS_NAMES)}


def generate_mq2_dataset(profiles, n_per_class=500, noise=0.1):
    """
    Generate synthetic MQ-2 sensor dataset from datasheet profiles.
    Features: ratio, ratio^2, 1/ratio, log(ratio), raw_adc_approx
    """
    X_list, y_list = [], []
    np.random.seed(42)

    for label, mean, std, _, class_id in profiles:
        n = n_per_class
        # Primary ratio reading (Rs/Ro)
        ratios = np.random.normal(mean, std, n)
        ratios = np.clip(ratios, 0.1, 25.0)  # physical limits

        # Engineered features from ratio
        ratio_sq  = ratios ** 2
        ratio_inv = 1.0 / ratios
        ratio_log = np.log(ratios)
        # Approximate raw ADC from ratio (inverse relationship)
        adc_approx = np.clip(1023 - (ratios / 15.0 * 1023), 0, 1023)
        # Add noise
        noise_arr = np.random.normal(0, noise, n)

        features = np.column_stack([
            ratios + noise_arr,
            ratio_sq,
            ratio_inv,
            ratio_log,
            adc_approx
        ])
        X_list.append(features)
        y_list.extend([class_id] * n)

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


print("=" * 50)
print("  Gas Model Training - MQ-2 Sensor")
print("=" * 50)

# ---- 1. Generate Dataset ----
print("\n[1/6] Generating MQ-2 synthetic dataset...")
X, y = generate_mq2_dataset(MQ2_GAS_PROFILES, n_per_class=600)
df = pd.DataFrame(X, columns=['ratio','ratio_sq','ratio_inv',
                                'ratio_log','adc_approx'])
df['label'] = [GAS_NAMES[i] for i in y]
print(f"  Total samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Class distribution:\n{df['label'].value_counts()}")

# ---- 2. EDA Plot ----
print("\n[2/6] Generating EDA plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['label'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Gas class distribution (before SMOTE)')
axes[0].set_xlabel('Gas type')
axes[0].tick_params(axis='x', rotation=30)

for name in GAS_NAMES:
    mask = df['label'] == name
    axes[1].hist(df[mask]['ratio'], alpha=0.6, label=name, bins=30)
axes[1].set_title('Rs/Ro ratio distribution per gas')
axes[1].set_xlabel('Rs/Ro ratio')
axes[1].legend()

plt.tight_layout()
plt.savefig('notebooks/plots/gas_eda.png', dpi=150)
print("  Saved: notebooks/plots/gas_eda.png")

# ---- 3. Train/Test Split ----
print("\n[3/6] Train/test split (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ---- 4. Scale + SMOTE ----
print("\n[4/6] Scaling + SMOTE oversampling...")
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
joblib.dump(scaler, 'models/gas_scaler.pkl')

sm = SMOTE(random_state=42)
X_bal, y_bal = sm.fit_resample(X_train, y_train)
print(f"  After SMOTE: {len(X_bal)} training samples")

# ---- 5. Train Random Forest ----
print("\n[5/6] Training Random Forest (100 trees)...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_bal, y_bal)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=GAS_NAMES))

# ---- 6. Save + Plots ----
print("\n[6/6] Saving model and generating evaluation plots...")
joblib.dump(rf, 'models/gas_rf_model.pkl')
joblib.dump(GAS_NAMES, 'models/gas_labels.pkl')
print("  Saved: models/gas_rf_model.pkl")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=GAS_NAMES, yticklabels=GAS_NAMES)
plt.title(f'Gas Classifier Confusion Matrix (Accuracy: {acc:.2%})')
plt.tight_layout()
plt.savefig('notebooks/plots/gas_confusion_matrix.png', dpi=150)
print("  Saved: notebooks/plots/gas_confusion_matrix.png")

# Feature importance
fi = pd.Series(rf.feature_importances_,
               index=['ratio','ratio_sq','ratio_inv',
                       'ratio_log','adc_approx'])
fi.sort_values().plot(kind='barh', figsize=(8, 4), color='steelblue')
plt.title('Feature Importance - Gas RF Model')
plt.tight_layout()
plt.savefig('notebooks/plots/gas_feature_importance.png', dpi=150)

print("\n  Training complete!")
print(f"  Accuracy: {acc*100:.2f}%")
print("\nNOTE: When you have hardware, collect real MQ-2 readings")
print("for each gas and retrain with actual sensor data for best accuracy.")
