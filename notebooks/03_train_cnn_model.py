"""
03_train_cnn_model.py
Train CNN (MobileNetV2) fire detection model.

Dataset:
    data/fire_dataset/fire/
    data/fire_dataset/no_fire/

Run:
    python notebooks/03_train_cnn_model.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import tensorflow as tf

os.makedirs("models", exist_ok=True)
os.makedirs("notebooks/plots", exist_ok=True)

DATASET_DIR = "data/fire_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif")

# ---- 0. Validate dataset ----
print("=" * 60)
print("  CNN Fire Detection Model Training")
print("=" * 60)

fire_dir = os.path.join(DATASET_DIR, "fire")
nofire_dir = os.path.join(DATASET_DIR, "no_fire")

if not os.path.isdir(fire_dir) or not os.path.isdir(nofire_dir):
    raise FileNotFoundError(
        f"Expected dataset folders: {fire_dir} and {nofire_dir}"
    )

fire_count = sum(
    1 for f in os.listdir(fire_dir)
    if f.lower().endswith(VALID_EXTENSIONS)
)
nofire_count = sum(
    1 for f in os.listdir(nofire_dir)
    if f.lower().endswith(VALID_EXTENSIONS)
)

print(f"\n  Fire images:    {fire_count}")
print(f"  No-fire images: {nofire_count}")
print(f"  Total images:   {fire_count + nofire_count}")

if fire_count < 50 or nofire_count < 50:
    raise ValueError("Not enough images in one or both classes.")

# ---- 1. Data generators with augmentation ----
print("\n[1/5] Setting up data generators...")

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    validation_split=0.2,
)

# Validation must NOT use random augmentation.
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

print(f"  Train: {train_data.samples} | Val: {val_data.samples}")
print(f"  Classes: {train_data.class_indices}")

# Balance the training classes without duplicating images.
classes = np.unique(train_data.classes)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_data.classes,
)
class_weights = dict(zip(classes, weights))
print(f"  Class weights: {class_weights}")

# ---- 2. Build MobileNetV2 transfer-learning model ----
print("\n[2/5] Building MobileNetV2 transfer learning model...")

base = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)
base.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

cnn = tf.keras.Model(inputs=base.input, outputs=out)
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(f"  Total params: {cnn.count_params():,}")

# ---- 3. Phase 1: Train classification head ----
print("\n[3/5] Phase 1: Training classification head...")

class StopAtAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("val_accuracy")
        if acc is not None and acc >= 0.92:
            print(f"\n  Target validation accuracy {acc:.4f} reached - stopping")
            self.model.stop_training = True

cb1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/cnn_fire_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    StopAtAccuracy(),
]

h1 = cnn.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=cb1,
)

# ---- 4. Phase 2: Fine-tune top MobileNetV2 layers ----
print("\n[4/5] Phase 2: Fine-tuning top 20 MobileNetV2 layers...")

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/cnn_fire_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
]

h2 = cnn.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    class_weight=class_weights,
    callbacks=callbacks_p2,
)

# ---- 5. Evaluate + save ----
print("\n[5/5] Evaluating and saving...")

cnn.save("models/cnn_fire_model.h5")
print("  Saved: models/cnn_fire_model.h5")

val_data.reset()
val_loss, val_acc = cnn.evaluate(val_data, verbose=0)
print(f"  Final val accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")

val_data.reset()
pred_prob = cnn.predict(val_data, verbose=0).ravel()
pred = (pred_prob >= 0.5).astype(int)
true = val_data.classes
class_names = [name for name, _ in sorted(
    val_data.class_indices.items(), key=lambda item: item[1]
)]

print("\n  Classification report:")
print(classification_report(true, pred, target_names=class_names, digits=4))

cm = confusion_matrix(true, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("notebooks/plots/cnn_confusion_matrix.png", dpi=150)
plt.close()

all_acc = h1.history["accuracy"] + h2.history["accuracy"]
all_val = h1.history["val_accuracy"] + h2.history["val_accuracy"]

plt.figure(figsize=(12, 4))
plt.plot(all_acc, label="Train accuracy")
plt.plot(all_val, label="Val accuracy")
plt.axvline(
    x=len(h1.history["accuracy"]) - 1,
    linestyle="--",
    label="Fine-tune start",
)
plt.title("CNN Training Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/plots/cnn_training_accuracy.png", dpi=150)
plt.close()

print("  Saved: notebooks/plots/cnn_training_accuracy.png")
print("  Saved: notebooks/plots/cnn_confusion_matrix.png")
print("\n  CNN training complete!")
