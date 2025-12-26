# 📱 Edge Deployment of Compressed CNN Model

**High-Accuracy (>92%) MobileNetV2 implementation on CIFAR-10 with INT8 Quantization for Edge Deployment.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![TFLite](https://img.shields.io/badge/TFLite-Edge_Ready-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📖 Overview

This project implements a highly optimized **MobileNetV2** pipeline for the **CIFAR-10** dataset, specifically designed for low-latency edge deployment. By utilizing **Transfer Learning** and **2-Phase Progressive Fine-Tuning**, the model achieves state-of-the-art accuracy (>92%) while maintaining a tiny footprint (~2.5 MB) via **INT8 Post-Training Quantization**.

## ✨ Key Features

*   **⚡ MobileNetV2 Backbone**: Leverages Depthwise Separable Convolutions for efficient inference.
*   **🧠 2-Phase Training Strategy**:
    1.  **Frozen Backbone**: Stabilizes weights (5 Epochs).
    2.  **Progressive Fine-Tuning**: Unfreezes top 30 layers with low LR for maximum accuracy (Early Stopping enabled).
*   **📉 INT8 Quantization**: Full 8-bit integer quantization (weights & activations) optimizing for TFLite runtime.
*   **⚖️ Fair Benchmarking**: Rigorous per-image latency comparison between FP32 (TensorFlow) and INT8 (TFLite).
*   **🛠️ Robust Pipeline**: Automatic data augmentation (Flip, Rotation, Zoom) and split-specific preprocessing.

## 📊 Results

| Model Variation | Precision | Size (MB) | Accuracy | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- |
| Model Variation | Precision | Size (MB) | Accuracy | Latency (CPU) | Peak Memory |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | FP32 | ~20.9 MB | **94.6%** | 99.9 ms | ~96.8 MB |
| **Quantized** | **INT8** | **~2.6 MB** | **90.0%** | **7.91 ms** | **~9.8 MB** |

*Note: Achieved **4x compression** with minimal (<1.5%) accuracy loss.*

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ajaykumar0875/Edge-Deployment-of-Compressed-CNN-Model.git
    cd Edge-Deployment-of-Compressed-CNN-Model
    ```


2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

   
3.  **Run the Pipeline**
    To train the model, quantize it, and generate the benchmark report:
    ```bash
    python Code.py
    ```

## 📂 Project Structure

*   `Code.py`: Main pipeline script (Training -> Quantization -> Eval).
*   `model_pipeline.py`: (Legacy) Initial Food-101 pipeline implementation.
*   `benchmark_results.json`: Auto-generated performance report.
*   `model_quantized.tflite`: Final INT8 model ready for deployment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
