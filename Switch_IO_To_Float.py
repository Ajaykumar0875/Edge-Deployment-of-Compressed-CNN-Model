
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

# Load Config
IMG_SIZE = 224
TRAIN_SAMPLES = 40000 
model_path = 'baseline_mobilenet_v2.h5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Cannot find {model_path}. Please run training first.")

print(f"Loading {model_path}...")
model = load_model(model_path)

# Prepare Data for Calibration
print("Loading Data for Calibration...")
(x_train, _), _ = cifar10.load_data()
x_train = x_train[:TRAIN_SAMPLES]

def representative_dataset_gen():
    indices = np.random.choice(len(x_train), 300, replace=False)
    for i in indices:
        img = x_train[i]
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img) # Normalize [-1, 1]
        img = tf.expand_dims(img, 0)
        yield [img]

print("Converting to TFLite (Float I/O, Int8 Weights)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# CRITICAL FIX: Set Interface to Float32 to match server.py
converter.inference_input_type = tf.float32  
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

output_path = 'model_quantized.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Success! Updated '{output_path}' to use Float32 Input.")
print("You can now restart server.py.")
