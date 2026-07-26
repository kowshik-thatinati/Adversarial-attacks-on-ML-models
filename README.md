# Adversarial Robustness of Object Detectors: An Empirical Study of YOLOv5 and Faster R-CNN Under Iterative FGSM Attacks

A Python framework for generating and evaluating adversarial examples against state-of-the-art object detection models (YOLOv5 and Faster R-CNN), using multi-step iterative FGSM (I-FGSM) perturbations. This project empirically studies how detector confidence and detection counts degrade under increasing attack strength, and compares vulnerability across a one-stage detector (YOLOv5) and a two-stage detector (Faster R-CNN).

This work extends research conducted during my internship on **Adversarial Robustness of Object Detectors** at IIT Tirupati (May 2025 – Aug 2025, under Dr. Chalavadi Vishnu), where I built the original iterative FGSM attack framework and benchmarking methodology. *[Edit this line to be precise about scope: state clearly whether this public repo is (a) the internship codebase itself, cleared for release, or (b) a personal extension/reimplementation built on the same methodology outside the internship's proprietary data/scope. Don't leave it ambiguous.]*

---

## 🎯 Project Overview

Modern object detectors achieve strong accuracy on benchmark datasets but remain vulnerable to small, often visually imperceptible input perturbations. This project:

1. Implements a multi-step iterative FGSM attack targeting detection confidence directly (not just classification loss)
2. Benchmarks attack effectiveness across two architecturally distinct detectors — a one-stage detector (YOLOv5) and a two-stage detector (Faster R-CNN) — to test whether architecture affects robustness
3. Quantifies the relationship between attack strength (epsilon) and confidence degradation via a systematic epsilon sweep
4. Tests **cross-model transferability** — whether examples crafted to fool one detector also fool the other, which speaks to whether the vulnerability is architecture-specific or a more general property of gradient-based detectors
5. Provides an interactive Gradio interface and CLI tools for reproducible, side-by-side inspection of clean vs. adversarial detections

**Key Features:**
- ✅ Real-time object detection with YOLOv5 and Faster R-CNN
- ✅ Multi-step iterative FGSM adversarial attack implementation
- ✅ Systematic epsilon-sweep methodology with averaged, reproducible metrics
- ✅ Cross-model transferability analysis
- ✅ Interactive web UI using Gradio
- ✅ CLI tools for batch processing
- ✅ Comprehensive visualization with bounding boxes
- ✅ GPU support (CUDA when available)

## 📋 Table of Contents
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Attack Details](#-attack-details)
- [Models](#-models)
- [Results](#-results)
- [Transferability Analysis](#-transferability-analysis)
- [Limitations](#-limitations)
- [Technical Specifications](#-technical-specifications)
- [Troubleshooting](#-troubleshooting)
- [References](#-references--papers)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA 11.8+ (optional, for GPU support)

### Step 1: Clone the Repository
```bash
git clone https://github.com/kowshik-thatinati/Adversarial-attacks-on-ML-models.git
cd Adversarial-attacks-on-ML-models
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models
The YOLOv5 model will be auto-downloaded on first run. The Faster R-CNN model is built from torchvision.

---

## ⚡ Quick Start

### Option 1: Web Interface (Recommended)
```bash
python app.py
```
The web interface will be available at `http://127.0.0.1:7860`

### Option 2: CLI - YOLOv5
```bash
python run_yolo.py --image custom_images/images.jpg --output results --epsilon 0.15
```

### Option 3: CLI - Faster R-CNN
```bash
python run_faster_rcnn.py --image custom_images/images.jpg --output results --epsilon 0.20
```

### Option 4: Reproduce the Epsilon Sweep / Transferability Study
```bash
python run_evaluation.py --dataset data/eval_set --epsilons 0.01 0.05 0.10 0.15 0.20 0.30 --transfer
```
*[This script doesn't exist yet in the current repo — see "Methodology" below. Build this as a batch runner that loops over your eval set and epsilon values and writes results to a CSV, so the tables below are regenerable rather than one-off.]*

---

## 📁 Project Structure
```
adversarial_yolo_project/
│
├── app.py                      # Main Gradio web interface
├── run_yolo.py                 # YOLOv5 CLI demonstration
├── run_faster_rcnn.py          # Faster R-CNN CLI demonstration
├── run_evaluation.py           # Batch epsilon-sweep + transferability runner [add this]
│
├── models/
│   ├── __init__.py
│   ├── yolo.py                 # YOLOv5 model wrapper
│   └── faster_rcnn.py          # Faster R-CNN model wrapper
│
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py                 # Legacy FGSM implementation
│   └── utils.py                # Attack utilities
│
├── attack_utils.py             # Multi-step I-FGSM attack implementation
├── inference_utils.py          # Inference helper functions
├── model_loader.py             # Model initialization
├── visualize.py                # Visualization utilities
│
├── custom_images/              # Sample images for testing
├── data/                       # COCO dataset annotations + eval subset
├── results/                    # Sweep outputs, CSVs, plots [add this]
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── QUICK_START.md              # Quick start guide
├── PROJECT_COMPLETION_SUMMARY.md
└── README.md                   # This file
```

---

## 🎮 Usage

### Web Interface (Gradio)

**Start the Application:**
```bash
python app.py
```

**Access the UI:** Open browser to `http://127.0.0.1:7860`

**Steps in Web UI:**
1. Select detection model (YOLO or Faster R-CNN)
2. Set attack strength (epsilon) — range: 0.01 to 0.50
3. Upload image
4. Click "Detect & Attack"
5. View original and adversarial detections side-by-side

**Parameters:**
- **Model:** Choose between YOLOv5 (faster) or Faster R-CNN (more accurate)
- **Epsilon:** Attack perturbation strength
  - 0.01–0.05: Subtle changes (minimal visual artifacts)
  - 0.05–0.15: Moderate changes (visible distortion)
  - 0.15–0.30: Strong changes (significant artifacts)
  - 0.30+: Extreme changes (heavy artifacts, poor visualization)

### Command Line Tools

**YOLOv5 CLI**
```bash
python run_yolo.py --image <image_path> --output <output_dir> --epsilon <float>
```

**Faster R-CNN CLI**
```bash
python run_faster_rcnn.py --image <image_path> --output <output_dir> --epsilon <float>
```

---

## 🔬 Methodology

*[This is the section that turns the repo from "a tool" into "a study." Fill in the bracketed values once you run the sweep — don't publish placeholders.]*

**Evaluation set:** [N] images sampled from [COCO val2017 subset / your custom_images set — specify which]. Images span [object count range, e.g. "3 to 30 annotated objects per image"] to test attack behavior across sparse and dense scenes.

**Attack configuration:** 10-step iterative FGSM, step size α = [value], evaluated at epsilon ∈ {0.01, 0.05, 0.10, 0.15, 0.20, 0.30} to characterize the full perturbation-strength curve rather than a single operating point.

**Metrics reported:**
- Mean detection confidence (averaged across all detected objects, across all N images, per epsilon)
- Standard deviation of confidence drop (to show consistency, not just a single anecdotal example)
- Detection count retention (objects detected post-attack ÷ objects detected pre-attack)
- Misclassification rate (% of retained detections with a changed class label)

**Why this matters:** the original results section reported single-image, single-run numbers (e.g. "0.693 → 0.243"). That's a valid illustrative example but not a benchmark. Averaging over a fixed evaluation set with reported variance is what makes the numbers below defensible rather than anecdotal.

---

## 🔥 Attack Details

### Multi-Step Iterative FGSM (I-FGSM)

The project implements a 10-step iterative FGSM attack that:
1. **Iteratively Perturbs:** Updates image gradients over 10 steps instead of a single step
2. **Region-Focused:** Targets detected object regions specifically
3. **Confidence Degradation:** Directly optimizes to reduce detection confidence scores
4. **Misclassification:** Occasionally causes class confusion (person → car, dog → cat)
5. **Generalization:** Works across different object sizes and positions

### Attack Algorithm
```
Input: Image I, Epsilon ε, Steps n=10
For each step i = 1 to n:
    1. Forward pass through detector
    2. Compute loss that decreases detection confidence
    3. Calculate gradients w.r.t. image
    4. Update image: I = I - α * sign(∇)
    5. Clip perturbation to [-ε, ε]
Output: Adversarial image I_adv
```

**Why iterative over single-step FGSM:** single-step FGSM takes one large gradient step, which is fast but coarse. I-FGSM takes several smaller steps, re-computing gradients at each step — this generally produces stronger, more precisely-targeted perturbations at the same epsilon budget. *[If you have single-step vs. iterative numbers, or can generate them, add a two-row comparison table here — it's a natural, cheap ablation and directly answers "why iterative."]*

---

## 🤖 Models

### 1. YOLOv5 (Ultralytics)
- **Architecture:** One-stage detector
- **Input Size:** 640×640 pixels
- **Classes:** 80 COCO classes
- **Inference Speed:** ~20–30ms per image (GPU)
- **Strengths:** Fast inference, good real-time performance
- **Weaknesses:** Slightly less accurate than Faster R-CNN

### 2. Faster R-CNN (ResNet-50)
- **Architecture:** Two-stage detector (region proposal + classification)
- **Input Size:** Variable (auto-resized)
- **Backbone:** ResNet-50
- **Classes:** 80 COCO classes
- **Inference Speed:** ~100–150ms per image (GPU)
- **Strengths:** Higher accuracy, robust to scale variations
- **Weaknesses:** Slower inference, heavier computation

**Why compare these two specifically:** one-stage and two-stage detectors differ fundamentally in how they generate and score candidate boxes. Comparing attack effectiveness across both tests whether adversarial vulnerability is a property of the gradient-based detection paradigm generally, or specific to a given architecture's design.

---

## 📊 Results

*[Replace every bracketed value below by running `run_evaluation.py` across your eval set. Do not publish estimated numbers — these need to be real, reproducible outputs.]*

### Epsilon Sweep — YOLOv5

| Epsilon (ε) | Mean Confidence (± std) | Detection Retention | Misclassification Rate |
|---|---|---|---|
| 0.00 (clean) | [x.xx ± x.xx] | 100% | — |
| 0.01 | [ ] | [ ] | [ ] |
| 0.05 | [ ] | [ ] | [ ] |
| 0.10 | [ ] | [ ] | [ ] |
| 0.15 | [ ] | [ ] | [ ] |
| 0.20 | [ ] | [ ] | [ ] |
| 0.30 | [ ] | [ ] | [ ] |

### Epsilon Sweep — Faster R-CNN

| Epsilon (ε) | Mean Confidence (± std) | Detection Retention | Misclassification Rate |
|---|---|---|---|
| 0.00 (clean) | [x.xx ± x.xx] | 100% | — |
| 0.01 | [ ] | [ ] | [ ] |
| 0.05 | [ ] | [ ] | [ ] |
| 0.10 | [ ] | [ ] | [ ] |
| 0.15 | [ ] | [ ] | [ ] |
| 0.20 | [ ] | [ ] | [ ] |
| 0.30 | [ ] | [ ] | [ ] |

*[Once filled in, a simple line plot of "mean confidence vs. epsilon" for both models on the same axes is worth adding as an image — it's the single most legible summary of this whole project and reads well in a portfolio/interview context.]*

### Illustrative Single-Image Example (YOLOv5, ε = 0.15)
```
Original Image Detections (9 objects):
  • Person: 0.693
  • Person: 0.657
  • Dog:    0.834
  • Cat:    0.712
  • ...

Adversarial Image (after attack, ε = 0.15):
  • Person: 0.243
  • Car:    0.457   (misclassified from Person)
  • Dog:    0.512   (confidence reduced)
  • ...
```
*(Kept as a qualitative illustration — the aggregate table above is the actual evidence; this example just shows what a single case looks like.)*

### Illustrative Single-Image Example (Faster R-CNN, ε = 0.20)
```
Original Detections (30 objects):
  • Person: 0.99
  • Person: 0.97
  • Dog:    0.95
  • ... (27 more)

Adversarial Image (after attack, ε = 0.20):
  • Person: 0.36
  • Dog:    0.42
  • ... (only 5-8 detections remain above threshold)
```

---

## 🔀 Transferability Analysis

*[This is the highest-value addition — a few hours of work with code you already have, and it's the difference between "I attacked two models" and "I studied whether adversarial vulnerability transfers across detector architectures," which is a genuine research question.]*

**Setup:** adversarial examples are crafted against a *source* model (white-box access to gradients) and then evaluated against the *target* model (no gradient access — this tests black-box transfer).

| Source → Target | Mean Confidence Drop (Target) | Detection Retention (Target) |
|---|---|---|
| YOLOv5 → YOLOv5 (white-box baseline) | [ ] | [ ] |
| YOLOv5 → Faster R-CNN (transfer) | [ ] | [ ] |
| Faster R-CNN → Faster R-CNN (white-box baseline) | [ ] | [ ] |
| Faster R-CNN → YOLOv5 (transfer) | [ ] | [ ] |

**Interpretation:** [Fill in once you have numbers. If transfer confidence drop is close to the white-box baseline, that suggests the vulnerability generalizes across architectures — a stronger and more concerning finding. If transfer drop is much smaller than white-box, that suggests attacks are largely architecture-specific, which is also a meaningful and reportable result. Either outcome is a real finding — don't discard this section if the numbers are "boring"; a null result honestly reported is still valid research.]

---

## ⚠️ Limitations

Being explicit about scope strengthens rather than weakens a research-style writeup:
- Evaluation restricted to COCO classes; results may not generalize to out-of-distribution object categories or domains (e.g., medical imaging, satellite imagery)
- I-FGSM is a relatively simple white-box attack; stronger attacks (PGD, C&W) were not benchmarked and may behave differently
- No defense/mitigation methods (adversarial training, input preprocessing) are evaluated — this project characterizes vulnerability, not robustness solutions
- [Add any dataset size or compute constraints that affected your evaluation set size]

---

## 🔧 Technical Specifications

### Dependencies
- **PyTorch:** 2.0+ (tensor operations, GPU support)
- **Ultralytics:** YOLOv5 implementation
- **Torchvision:** Faster R-CNN, image transforms
- **Gradio:** 4.0+ (web interface)
- **OpenCV:** Image processing
- **Matplotlib:** Visualization
- **NumPy:** Numerical operations
- **Pillow:** Image loading

### Hardware Requirements

**Minimum:**
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Disk: 2 GB (models auto-download)

**Recommended (GPU):**
- GPU: NVIDIA with CUDA 11.8+
- VRAM: 4–6 GB
- RAM: 16 GB
- Disk: 3 GB

### Software Requirements
- Python: 3.8, 3.9, 3.10, 3.11
- OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+

---

## 🐛 Troubleshooting

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: "CUDA out of memory"**

Use CPU instead:
```python
# In app.py or other files, modify device initialization
device = torch.device('cpu')
```

**Issue: "Web UI shows wrong detections"**

Make sure you're using the exact same `YOLOModel` class from `models/yolo.py` in `app.py`.

**Issue: "Gradio interface not accessible"**
```python
# Check if port 7860 is available
# If occupied, modify in app.py:
# interface.launch(share=True, server_port=7861)
```

**Issue: "Model download fails"**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov5su.pt')"
```

---

## 📚 References & Papers
- **FGSM Attack:** Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
- **YOLOv5:** Ultralytics, https://github.com/ultralytics/yolov5
- **Faster R-CNN:** Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (2015)
- **Adversarial ML:** Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2016)
- **Iterative Attacks:** Kurakin et al., "Adversarial Examples in the Physical World" (2016) — the basis for the multi-step I-FGSM approach used here *[add if you drew on this or an equivalent paper; cite whatever your internship work actually referenced]*

---

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

**Contributing Guidelines:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License. See LICENSE file for details.

## 👤 Author
**Kowshik Thatinati**
- GitHub: [@kowshik-thatinati](https://github.com/kowshik-thatinati)
- LinkedIn: [linkedin.com/in/kowshik-thatinati](https://linkedin.com/in/kowshik-thatinati)
- Email: kowshikthatinati559@gmail.com

## 🎓 Research Context

This project was developed as an extension of research conducted during my Adversarial Robustness of Object Detectors internship at IIT Tirupati (May–Aug 2025, supervised by Dr. Chalavadi Vishnu), where the original iterative FGSM framework and confidence-degradation methodology were developed and applied across YOLOv5 and Faster R-CNN detectors, achieving 60–85% confidence degradation under fixed perturbation budgets. This public repository documents the methodology and provides a reproducible benchmarking framework for further study.

*Disclaimer: This tool is intended for research and educational purposes — to study and demonstrate model vulnerabilities — and should only be used on systems you own or have explicit permission to test.*

## 📞 Support
For issues, questions, or suggestions:
1. Check the Troubleshooting section above
2. Search existing GitHub issues
3. Open a new GitHub issue with a detailed description

## 🎉 Acknowledgments
- Ultralytics for YOLOv5
- Facebook Research (Meta AI) for Faster R-CNN
- PyTorch Foundation for the deep learning framework
- Gradio team for the web interface library
- Dr. Chalavadi Vishnu, IIT Tirupati, for research supervision

---

**Last Updated:** [update on publish date]
**Project Status:** ✅ Core framework complete — evaluation/transferability study in progress
