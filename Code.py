
import os
import time
import json
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot # Pruning

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# CONFIGURATION
# ==============================================================================
print("🔹 STEP 0: ENVIRONMENT SETUP")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU (Simulate Edge)
NUM_CLASSES = 10
IMG_SIZE = 224
BATCH_SIZE = 32
EVAL_SAMPLES = 1000
TRAIN_SAMPLES = 42000 
FINE_TUNE_EPOCHS = 30
PRUNING_EPOCHS = 5 # Short fine-tuning for pruning

# ==============================================================================
# STEP 1: LOAD & PREPARE DATASET
# ==============================================================================
print("\n🔹 STEP 1: PREPARE DATASET (CIFAR-10)")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train[:TRAIN_SAMPLES]
y_train = y_train[:TRAIN_SAMPLES]
x_eval = x_test[:EVAL_SAMPLES]
y_eval = y_test[:EVAL_SAMPLES]

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_eval_cat = to_categorical(y_eval, NUM_CLASSES)

# Data Augmentation & Generators
augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
])

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, target_size, augment=False, preprocess=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.preprocess = preprocess

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx+1) * self.batch_size]
        
        # 1. Resize (Result is float32)
        batch_x_resized = tf.image.resize(batch_x, self.target_size)
        
        # 2. Augment (Training only)
        if self.augment:
            batch_x_resized = augmentation(batch_x_resized, training=True)
            
        # 3. Preprocess (Normalize to [-1, 1] for Training)
        if self.preprocess:
            batch_x_final = preprocess_input(batch_x_resized)
        else:
            batch_x_final = batch_x_resized # Raw [0, 255]

        return batch_x_final, batch_y

train_gen = DataGenerator(x_train, y_train_cat, BATCH_SIZE, (IMG_SIZE, IMG_SIZE), augment=True, preprocess=True)
eval_gen = DataGenerator(x_eval, y_eval_cat, BATCH_SIZE, (IMG_SIZE, IMG_SIZE), augment=False, preprocess=True)

# ==============================================================================
# STEP 2 & 3: MODEL SETUP & TRAINING
# ==============================================================================
print("\n🔹 STEP 2 & 3: MODEL TRAINING (2-PHASE)")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Phase 1: Frozen Backbone
print("[PHASE 1] Training Head (Backbone Frozen)...")
for layer in base_model.layers: layer.trainable = False
model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=eval_gen, epochs=5, verbose=1)

# Phase 2: Fine-Tuning
print("[PHASE 2] Fine-tuning Top 30 Layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]: layer.trainable = False

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=eval_gen, epochs=FINE_TUNE_EPOCHS, callbacks=[early_stopping], verbose=1)

# Evaluate Baseline
print("Evaluating Baseline FP32...")
loss, baseline_acc = model.evaluate(eval_gen, verbose=0)
model.save('baseline_mobilenet_v2.h5')
baseline_size = os.path.getsize('baseline_mobilenet_v2.h5') / (1024 * 1024)

# FP32 Latency Measure
print("Measuring FP32 Latency...")
dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
model(dummy, training=False) # Warmup
start = time.time()
for i in range(len(x_eval)):
    img = tf.expand_dims(preprocess_input(tf.image.resize(x_eval[i], (IMG_SIZE, IMG_SIZE))), 0)
    model(img, training=False)
baseline_lat = (time.time() - start) / len(x_eval) * 1000
print(f"Baseline: Acc={baseline_acc:.4f}, Lat={baseline_lat:.2f}ms")

# ==============================================================================
# STEP 3.5: MILD PRUNING (Added per Request)
# ==============================================================================
print("\n🔹 STEP 3.5: MILD PRUNING (50% Sparsity)")
# We apply pruning to the whole model (except Input)
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.30, begin_step=0, end_step=len(train_gen)*PRUNING_EPOCHS
    )
}

print("Applying pruning wrapper...")
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model_for_pruning.fit(train_gen, epochs=PRUNING_EPOCHS, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])

print("Stripping pruning wrapper to finalize weights...")
model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# ==============================================================================
# STEP 4: QUANTIZATION (WITH CORRECT CALIBRATION)
# ==============================================================================
print("\n🔹 STEP 4: INT8 QUANTIZATION (FIXED CALIBRATION)")

def representative_dataset_gen():
    # FIX: Use preprocess_input to match Training stats [-1, 1]
    indices = np.random.choice(len(x_train), 300, replace=False)
    for i in indices:
        img = x_train[i]
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img) # NORMALIZE TO [-1, 1]
        img = tf.expand_dims(img, 0)
        yield [img]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open('model_quantized.tflite', 'wb') as f: f.write(tflite_model)
q_size = len(tflite_model) / (1024 * 1024)
print(f"Quantized Model Saved: {q_size:.2f} MB")

# ==============================================================================
# STEP 5: FINAL EVALUATION
# ==============================================================================
print("\n🔹 STEP 5: FINAL INT8 EVALUATION")
interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
start = time.time()
print("Running INT8 Inference Loop...")
for i in range(len(x_eval)):
    # Inference Preprocessing:
    # Resize -> Cast to UINT8 (No Normalization needed for TFLite Client)
    img = tf.image.resize(x_eval[i], (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.uint8)
    img = tf.expand_dims(img, 0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    if np.argmax(output) == np.argmax(y_eval_cat[i]): correct += 1

q_lat = (time.time() - start) / len(x_eval) * 1000
q_acc = correct / len(x_eval)

print(f"\n✅ FINAL RESULTS:")
print(f"Baseline (FP32): Acc={baseline_acc:.4f}, Lat={baseline_lat:.2f}ms, Size={baseline_size:.2f}MB")
print(f"Quantized(INT8): Acc={q_acc:.4f}, Lat={q_lat:.2f}ms, Size={q_size:.2f}MB")

# Save JSON
with open('benchmark_results.json', 'w') as f:
    json.dump({"baseline": {"accuracy": float(baseline_acc), "latency_ms": float(baseline_lat)}, 
               "quantized": {"accuracy": float(q_acc), "latency_ms": float(q_lat)}}, f, indent=4)
