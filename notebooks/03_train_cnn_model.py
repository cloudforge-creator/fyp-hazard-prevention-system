"""
03_train_cnn_model.py
Train CNN (MobileNetV2) fire detection model.

DATASET SETUP:
1. Download from Kaggle:
   - "Fire Detection Dataset" by phylake1337
   - "The Wildfire Dataset" by elmadafri
2. Organise images into:
   data/fire_images/fire/      <- all fire and smoke images
   data/fire_images/no_fire/   <- all normal images

Run: python notebooks/03_train_cnn_model.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

os.makedirs('models', exist_ok=True)
os.makedirs('notebooks/plots', exist_ok=True)

FIRE_DIR    = 'data/fire_images/fire'
NOFIRE_DIR  = 'data/fire_images/no_fire'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

# Check dataset exists
fire_count   = len([f for f in os.listdir(FIRE_DIR)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(FIRE_DIR) else 0
nofire_count = len([f for f in os.listdir(NOFIRE_DIR)
                    if f.lower().endswith(('.jpg','.jpeg','.png'))]) \
               if os.path.exists(NOFIRE_DIR) else 0

print("=" * 50)
print("  CNN Fire Detection Model Training")
print("=" * 50)
print(f"\n  Fire images:    {fire_count}")
print(f"  No-fire images: {nofire_count}")

if fire_count < 50 or nofire_count < 50:
    print("\n  Not enough images. Please download datasets from Kaggle:")
    print("  1. https://www.kaggle.com/datasets/phylake1337/fire-dataset")
    print("  2. Place fire images in:    data/fire_images/fire/")
    print("  3. Place no-fire images in: data/fire_images/no_fire/")
    print("\n  Minimum: 100 images per class (500+ recommended)")
    exit(0)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D,
                                      Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)

# ---- 1. Data Generators with Augmentation ----
print("\n[1/5] Setting up data generators with augmentation...")
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    validation_split=0.2
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'data/fire_images/', target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary', subset='training')
val_data = train_gen.flow_from_directory(
    'data/fire_images/', target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary', subset='validation')
print(f"  Train: {train_data.samples} | Val: {val_data.samples}")
print(f"  Classes: {train_data.class_indices}")

# ---- 2. Build Model with MobileNetV2 ----
print("\n[2/5] Building MobileNetV2 transfer learning model...")
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(*IMG_SIZE, 3))
base.trainable = False  # freeze initially

x   = GlobalAveragePooling2D()(base.output)
x   = BatchNormalization()(x)
x   = Dense(256, activation='relu')(x)
x   = Dropout(0.4)(x)
x   = Dense(64, activation='relu')(x)
x   = Dropout(0.2)(x)
out = Dense(1, activation='sigmoid')(x)

cnn = Model(inputs=base.input, outputs=out)
cnn.compile(optimizer=Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy'])
print(f"  Trainable params: {cnn.count_params():,}")

# ---- 3. Phase 1: Train top layers (frozen base) ----
print("\n[3/5] Phase 1: Training classification head (10 epochs)...")
callbacks_p1 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True),
    ModelCheckpoint('models/cnn_fire_best.h5',
                    monitor='val_accuracy', save_best_only=True)
]
h1 = cnn.fit(train_data, epochs=10,
             validation_data=val_data, callbacks=callbacks_p1)

# ---- 4. Phase 2: Fine-tune top layers of MobileNetV2 ----
print("\n[4/5] Phase 2: Fine-tuning top 20 MobileNetV2 layers...")
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

cnn.compile(optimizer=Adam(1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy'])
callbacks_p2 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True),
]
h2 = cnn.fit(train_data, epochs=10,
             validation_data=val_data, callbacks=callbacks_p2)

# ---- 5. Evaluate + Save ----
print("\n[5/5] Evaluating and saving...")
cnn.save('models/cnn_fire_model.h5')
print("  Saved: models/cnn_fire_model.h5")

val_loss, val_acc = cnn.evaluate(val_data, verbose=0)
print(f"  Final val accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

# Training curve
plt.figure(figsize=(12, 4))
all_acc = h1.history['accuracy'] + h2.history['accuracy']
all_val = h1.history['val_accuracy'] + h2.history['val_accuracy']
plt.plot(all_acc, label='Train accuracy')
plt.plot(all_val, label='Val accuracy')
plt.axvline(x=len(h1.history['accuracy'])-1,
            color='red', linestyle='--', label='Fine-tune start')
plt.title('CNN Training Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('notebooks/plots/cnn_training_accuracy.png', dpi=150)
print("  Saved: notebooks/plots/cnn_training_accuracy.png")
print("\n  CNN training complete!")
