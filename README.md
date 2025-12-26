# Adversarial Attacks on Object Detection Models  
### YOLOv5 and Swin Transformer

## Problem Statement
Modern object detection models such as YOLO and Transformer-based detectors are widely deployed in safety-critical applications including autonomous driving, surveillance, and intelligent transportation systems. These models are typically evaluated on clean data and assumed to be reliable under real-world conditions.

However, recent studies have shown that even **small, carefully crafted perturbations** to input images can cause object detection models to:
- Miss objects completely
- Drastically reduce detection confidence
- Misclassify detected objects

This raises serious concerns about the **robustness and reliability** of object detection systems, especially when adversarial manipulation is localized and visually imperceptible.

The core problem addressed in this project is:
> *How vulnerable are state-of-the-art object detection models to localized adversarial attacks applied only on detected object regions, rather than the entire image?*

---

## Project Theme
This project focuses on **adversarial robustness analysis of object detection models**, with an emphasis on **region-based adversarial attacks**.

Instead of applying adversarial noise globally across the entire image, the project explores a more realistic and targeted threat model where:
- Adversarial perturbations are applied **only inside object bounding boxes**
- Attacks are generated using gradient-based methods (FGSM)
- The effect of attacks is analyzed by comparing detections **before and after perturbation**

The study is conducted on:
- **YOLOv5** – a real-time, CNN-based object detector
- **Swin Transformer** – a Transformer-based vision model with hierarchical attention

The theme bridges **computer vision, deep learning security, and model robustness**, aiming to understand whether modern detection architectures—CNNs and Transformers alike—are resilient to localized adversarial manipulation.

This work serves as a foundational step toward:
- Robust object detection
- Adversarial defense mechanisms
- Security-aware deployment of vision models in real-world systems
