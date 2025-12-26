
import os
import time
import psutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Config
IMG_SIZE = 224
process = psutil.Process(os.getpid())

def get_memory_usage():
    """Returns current process memory usage in MB"""
    return process.memory_info().rss / (1024 * 1024)

print(f"Initial Memory: {get_memory_usage():.2f} MB")

# ==============================================================================
# 1. MEASURE FP32 MEMORY (Weights + Inference)
# ==============================================================================
print("\n🔹 MEASURING FP32 MEMORY...")
mem_before = get_memory_usage()

# Load Model
model = load_model('baseline_mobilenet_v2.h5')
dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))

# Run Inference Loop (simulate load)
for _ in range(50):
    model(dummy_input, training=False)

mem_after = get_memory_usage()
fp32_mem_peak = mem_after - mem_before
print(f"FP32 Peak Memory Usage: {fp32_mem_peak:.2f} MB")

# Cleanup to clear memory for next test
del model
tf.keras.backend.clear_session()
import gc
gc.collect()
time.sleep(2) # Let OS reclaim

# ==============================================================================
# 2. MEASURE INT8 TFLITE MEMORY
# ==============================================================================
print("\n🔹 MEASURING INT8 MEMORY...")
mem_before = get_memory_usage()

interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_idx = input_details[0]['index']

# Dummy Input (UINT8)
dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

# Run Inference Loop
for _ in range(50):
    interpreter.set_tensor(input_idx, dummy_input)
    interpreter.invoke()

mem_after = get_memory_usage()
int8_mem_peak = mem_after - mem_before
print(f"INT8 Peak Memory Usage: {int8_mem_peak:.2f} MB")

print(f"\n✅ FINAL MEMORY STATS:")
print(f"FP32 Usage: ~{fp32_mem_peak:.2f} MB")
print(f"INT8 Usage: ~{int8_mem_peak:.2f} MB ({(fp32_mem_peak/int8_mem_peak):.1f}x Lower)")
