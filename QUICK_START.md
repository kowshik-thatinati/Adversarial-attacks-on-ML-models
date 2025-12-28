# 🚀 Quick Start Guide - Adversarial YOLO Project

## Installation & Setup (One-Time)

```bash
# Navigate to project directory
cd c:\Users\desktop_files\Desktop\adversarial_yolo_project

# Install dependencies (if not already done)
pip install -r requirements.txt

# Ensure you have:
# - PyTorch 2.0+
# - Ultralytics YOLOv5
# - Torchvision
# - Gradio 4.0+
```

---

## 🎯 Run Web Interface (RECOMMENDED)

```bash
# Start the Gradio web app
python app.py

# Open browser: http://127.0.0.1:7860
```

**Steps in Web UI:**
1. Select model: **YOLOv5** or **Faster R-CNN**
2. Upload image (jpg/png format)
3. Set attack strength (epsilon): **0.01-0.20** (default 0.15)
4. Select attack mode: **entire_image** or **bounding_boxes_only**
5. Click **"Run Attack"**
6. View side-by-side comparison:
   - **Left**: Original detections with high confidence
   - **Right**: Adversarial detections with reduced confidence + occasional misclassification
7. Download results if needed

---

## 🎬 CLI Examples

### YOLO Attack (Quick Test)
```bash
python run_yolo.py --image custom_images/images.jpg --epsilon 0.15
```

**Output:**
```
Found 9 objects in original image
Original detections:
  person: 0.693
  person: 0.657
  traffic light: 0.442

Adversarial detections (reduced confidence & misclassification):
  car: 0.243      # Changed from person!
  person: 0.457   # Confidence reduced
  traffic light: 0.242  # Significantly reduced
```

### Faster R-CNN Attack
```bash
python run_faster_rcnn.py --image custom_images/images.jpg --epsilon 0.20
```

**Output:**
```
Found 30 objects in original image (confidence 0.99+)
Found 5 objects in adversarial image (confidence 0.24-0.63)
Attack Success: 83% detection reduction!
```

---

## 📊 Understanding the Results

### Confidence Score Changes
```
Original:  0.693  →  Adversarial: 0.243  (65% drop) ✅
Original:  0.657  →  Adversarial: 0.457  (30% drop) ✅
Original:  0.442  →  Adversarial: 0.242  (45% drop) ✅
```

### Misclassification Example
```
Original:  person (0.693)  →  Adversarial: car (0.243) ✅
           Same object, but model predicts different class!
```

### Why This Matters
- **Model Vulnerability**: Deep learning models are easily fooled by subtle pixel changes
- **Imperceptible Attack**: Humans can't see the difference, but model output changes dramatically
- **Real-World Risk**: Shows importance of adversarial robustness in autonomous systems

---

## 🎨 Visualization Features

### Bounding Boxes
- ✅ **Green boxes**: Original detections (correctly identified)
- ✅ **Red boxes**: Adversarial detections (confused predictions)
- ✅ **All labels inside image**: No floating text
- ✅ **Confidence scores**: Shown next to each box

### Comparison Layout
```
┌─────────────────────────────────────────┐
│   Original Detections (YOLOv5)          │
│   ┌──────────┐                          │
│   │ person   │ 0.693                    │
│   │ 0.693    │                          │
│   └──────────┘                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│   Adversarial Detections (YOLOv5)       │
│   ┌──────────┐                          │
│   │ car      │ 0.243  (misclassified!)  │
│   │ 0.243    │                          │
│   └──────────┘                          │
└─────────────────────────────────────────┘
```

---

## ⚙️ Parameter Guide

### Epsilon (Attack Strength)
- **0.01-0.05**: Weak attack, minor confidence drop
- **0.05-0.10**: Moderate attack, noticeable effect
- **0.10-0.20**: Strong attack, significant misclassification
- **Default**: 0.15 (good balance)

### Attack Mode
- **entire_image**: Perturb whole image (stronger effect)
- **bounding_boxes_only**: Only perturb detected objects (subtle)

### Model Selection
- **YOLOv5**: Fast (50-100ms), 80 COCO classes
- **Faster R-CNN**: Accurate (200-300ms), 80 COCO classes

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| App won't start | Check port 7860 is free: `netstat -ano \| findstr :7860` |
| YOLO slow on first run | Model downloads (~80MB), subsequent runs are fast |
| "Out of memory" error | Use CPU mode or smaller images |
| Detections look wrong | Try with provided test image first |
| Labels showing "background" | Ensure you're using web UI (fixed in latest version) |

---

## 📁 Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Main Gradio web interface |
| `run_yolo.py` | YOLO CLI demonstration |
| `run_faster_rcnn.py` | Faster R-CNN CLI demonstration |
| `models/yolo.py` | YOLOv5 wrapper class |
| `attack_utils.py` | FGSM adversarial attack implementation |
| `visualize.py` | Bounding box visualization |
| `custom_images/images.jpg` | Test image |
| `results/` | Output directory for generated images |

---

## 🎓 What You'll Learn

This project demonstrates:

1. **Adversarial Machine Learning**: How to fool deep learning models
2. **FGSM Attack**: Fast Gradient Sign Method adversarial perturbation
3. **Model Vulnerabilities**: Why even robust models can fail
4. **Defense Importance**: Why adversarial robustness matters
5. **Object Detection**: How YOLO and Faster R-CNN work
6. **Confidence Collapse**: How models become uncertain under attack

---

## 📈 Expected Results

### YOLO (Baseline)
- Original: 9 objects detected (all correct)
- After Attack: 9 objects with 40-50% confidence drop
- Misclassification: ~25% change predicted class

### Faster R-CNN (Baseline)
- Original: 30 objects detected (many false positives)
- After Attack: Only 5 objects (83% drop in detections)
- Shows attack is more effective on sensitive models

---

## 🎯 Next Steps

1. **Test with your own images**: Use any jpg/png image
2. **Try different epsilon values**: See how strength affects results
3. **Compare models**: See which is more robust
4. **Understand the attack**: Read the code in `attack_utils.py`
5. **Explore defenses**: Consider how to make models more robust

---

## 💡 Tips for Best Results

- Use images with clear objects (people, cars, traffic lights)
- Start with epsilon=0.15 for good balance
- Use "entire_image" mode for stronger attack effect
- Try both YOLO and Faster R-CNN to compare robustness
- Download the comparison images for documentation

---

## 📝 Example Workflow

```
1. Start app: python app.py
2. Open: http://127.0.0.1:7860
3. Select: YOLOv5
4. Upload: custom_images/images.jpg
5. Set epsilon: 0.15
6. Select mode: entire_image
7. Click: "Run Attack"
8. View: Side-by-side comparison
9. Note: Confidence changes and misclassification
10. Download: Results for presentation
```

---

## 🎉 You're Ready!

The project is fully functional. Start with the web UI for easiest interaction, or use CLI for detailed output.

**Happy adversarial learning!** 🚀

---

*Last Updated: December 28, 2025*
