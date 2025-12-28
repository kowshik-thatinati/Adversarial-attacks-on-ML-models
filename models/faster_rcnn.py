import torch
import torchvision
import cv2
import numpy as np

class FasterRCNNModel:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        self.model.eval()
        self.model.to(self.device)
        
        # COCO class names
        self.classes = [
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
    
    def preprocess(self, image_path):
        """Load and preprocess image for Faster R-CNN"""
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to tensor in [0, 1]. Torchvision detection models normalize internally.
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        img_tensor = transform(img).unsqueeze(0).to(self.device)
        return img_tensor, img
    
    def predict(self, img_tensor):
        """Run Faster R-CNN prediction"""
        with torch.no_grad():
            predictions = self.model([img_tensor.squeeze(0)])
        return predictions
    
    def process_image(self, image_path, conf_threshold=0.5):
        """Process image and return detections"""
        img_tensor, original_img = self.preprocess(image_path)
        predictions = self.predict(img_tensor)
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Format as [x1, y1, x2, y2, conf, class_id]
        if len(boxes) > 0:
            detections = np.column_stack([
                boxes,
                scores,
                labels.astype(float)
            ])
            detections = torch.tensor(detections, device=self.device)
        else:
            detections = torch.empty((0, 6), device=self.device)
            
        return detections, original_img, img_tensor
