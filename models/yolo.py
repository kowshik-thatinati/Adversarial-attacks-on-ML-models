import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO('yolov5s.pt')
        self.model.to(self.device)
        
    def preprocess(self, image_path):
        """Load and preprocess image for YOLO"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (640, 640))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img_tensor, img_resized  # Return resized image so coordinates match
    
    def predict(self, img_tensor):
        """Run YOLO prediction"""
        with torch.no_grad():
            # Convert tensor back to numpy for ultralytics
            img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            results = self.model(img_np, verbose=False)  # Suppress verbose output
            # Convert results to tensor format
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy
                scores = results[0].boxes.conf
                classes = results[0].boxes.cls
                # Combine into [x1, y1, x2, y2, conf, cls] format
                predictions = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
            else:
                predictions = torch.empty((0, 6))
        return predictions
    
    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45):
        """YOLOv5 non-max suppression - not needed for ultralytics"""
        # Ultralytics already applies NMS, so return the prediction as is
        if isinstance(prediction, list):
            return prediction
        return [prediction]
    
    @staticmethod
    def xywh2xyxy(x):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y
    
    def process_image(self, image_path):
        """Process image and return detections"""
        img_tensor, original_img = self.preprocess(image_path)
        predictions = self.predict(img_tensor)
        # The ultralytics model already applies NMS, so we can use predictions directly
        detections = predictions
        return detections, original_img, img_tensor
