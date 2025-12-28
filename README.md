# Adversarial Object Detection

This project demonstrates adversarial attacks on object detection models (YOLOv5 and Faster R-CNN) using the FGSM attack.

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### YOLOv5
```bash
python run_yolo.py --image path/to/image.jpg --output results --epsilon 0.05
```

### Faster R-CNN
```bash
python run_faster_rcnn.py --image path/to/image.jpg --output results --epsilon 0.05
```

## Arguments
- `--image`: Path to input image
- `--output`: Output directory (default: 'results')
- `--epsilon`: Attack strength (default: 0.05)
