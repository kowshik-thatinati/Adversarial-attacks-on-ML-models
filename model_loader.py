import torch
import torchvision
import cv2
import numpy as np
from ultralytics import YOLO

class ModelLoader:
    """Load and manage object detection models"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.current_model = None
        self.current_model_name = None
        
        # COCO class names
        self.coco_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def load_yolo(self):
        """Load YOLOv5 model - using medium for better accuracy"""
        if 'yolo' not in self.models:
            print("Loading YOLOv5 model (medium variant for high accuracy)...")
            try:
                # Try loading from local file first
                model = YOLO('yolov5su.pt')
                print("Loaded local yolov5su.pt")
            except:
                try:
                    # Try medium model for better accuracy
                    print("Local model not found, downloading YOLOv5m (medium)...")
                    model = YOLO('yolov5m.pt')
                except:
                    # Fall back to small model
                    print("Downloading YOLOv5s (small)...")
                    model = YOLO('yolov5s.pt')
            
            # Set to eval mode for inference
            model.model.eval()
            
            # Move to correct device
            if torch.cuda.is_available():
                model.model.to(self.device)
                print("YOLO model moved to GPU")
            else:
                print("YOLO model using CPU")
            
            self.models['yolo'] = model
        return self.models['yolo']
    
    def load_faster_rcnn(self):
        """Load Faster R-CNN model"""
        if 'faster_rcnn' not in self.models:
            print("Loading Faster R-CNN model...")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
            model.eval()
            model.to(self.device)
            self.models['faster_rcnn'] = model
        return self.models['faster_rcnn']
    
    def set_model(self, model_name):
        """Set the current model"""
        if model_name == 'YOLOv5':
            self.current_model = self.load_yolo()
            self.current_model_name = 'yolo'
        elif model_name == 'Faster R-CNN':
            self.current_model = self.load_faster_rcnn()
            self.current_model_name = 'faster_rcnn'
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_current_model(self):
        """Get the current loaded model"""
        return self.current_model, self.current_model_name
    
    def get_class_names(self):
        """Get COCO class names"""
        return self.coco_classes
