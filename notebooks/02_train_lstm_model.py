"""
03_train_cnn_model.py - LITE VERSION for laptops
Reduced image size (128x128), smaller batch, limited samples.
Accuracy slightly lower but trains without crashing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks/plots', exist_ok=True)

FIRE_DIR   = 'data/fire_images/fire'
NOFIRE_DIR = 'data/fire_images/no_fire'
IMG_SIZE   = (128, 128)   # reduced from 224 - saves RAM
BATCH_SIZE = 16           # reduced from 32 - saves RAM
MAX_PER_CLASS = 3000      # use max 3000 per class - prevents crash

fire_count   = len([f for f in os.listdir(FIRE_DIR)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(FIRE_DIR) else 0
nofire_count = len([f for f in os.listdir(NOFIRE_DIR)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(NOFIRE_DIR) else 0

print("=" * 50)
print("  CNN Fire Detection Model Training (LITE)")
print("=" * 50)
print(f"\n  Fire images:    {fire_count}")
print(f"  No-fire images: {nofire_count}")
print(f"  Using max {MAX_PER_CLASS} per class to prevent RAM crash")

if fire_count < 100 or nofire_count < 100:
    print("\n  Not enough images. Need 100+ per class.")
    exit(0)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Limit TensorFlow memory growth - prevents crash
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Limit CPU memory usage
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

print("\n[1/5] Setting up data generators...")

# Use fewer images - sample MAX_PER_CLASS from each folder
use_count = min(fire_count, nofire_count, MAX_PER_CLASS)
print(f"  Using {use_count} images per class = {use_count*2} total")

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.15,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    'data/fire_images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=42
)
val_data = train_gen.flow_from_directory(
    'data/fire_images/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"  Train batches: {len(train_data)} | Val batches: {len(val_data)}")
print(f"  Classes: {train_data.class_indices}")

print("\n[2/5] Building MobileNetV2 model (128x128 input)...")
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base.trainable = False

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(128, activation='relu')(x)
x   = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

cnn = Model(inputs=base.input, outputs=out)
cnn.compile(
    optimizer=Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print(f"  Trainable params: {sum(p.numpy().size for p in cnn.trainable_weights):,}")

print("\n[3/5] Phase 1 - Training top layers (5 epochs)...")
cb1 = [
    EarlyStopping(monitor='val_accuracy', patience=3,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('models/cnn_fire_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True, verbose=0)
]
h1 = cnn.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=cb1,
    verbose=1
)

print("\n[4/5] Phase 2 - Fine-tuning top 15 layers (5 epochs)...")
base.trainable = True
for layer in base.layers[:-15]:
    layer.trainable = False

cnn.compile(
    optimizer=Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
cb2 = [
    EarlyStopping(monitor='val_accuracy', patience=3,
                  restore_best_weights=True, verbose=1)
]
h2 = cnn.fit(
    train_data,
    epochs=5,
    validation_data=val_data,
    callbacks=cb2,
    verbose=1
)

print("\n[5/5] Saving model and plots...")
cnn.save('models/cnn_fire_model.h5')
print("  Saved: models/cnn_fire_model.h5")

val_loss, val_acc = cnn.evaluate(val_data, verbose=0)
print(f"  Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# Training curve plot
all_acc = h1.history['accuracy'] + h2.history['accuracy']
all_val = h1.history['val_accuracy'] + h2.history['val_accuracy']
plt.figure(figsize=(10, 4))
plt.plot(all_acc, label='Train accuracy', marker='o')
plt.plot(all_val, label='Val accuracy', marker='s')
plt.axvline(x=len(h1.history['accuracy'])-1,
            color='red', linestyle='--', label='Fine-tune start')
plt.title(f'CNN Training - Final Val Accuracy: {val_acc:.2%}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('notebooks/plots/cnn_training_accuracy.png', dpi=150)
print("  Saved: notebooks/plots/cnn_training_accuracy.png")

print("\n  CNN training complete!")
print(f"  Model saved at: models/cnn_fire_model.h5")
print(f"  Accuracy: {val_acc*100:.2f}%")