import torch
import torchvision
import cv2
import numpy as np
from ultralytics import YOLO

class InferenceUtils:
    """Utilities for model inference and detection processing"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for inference - robust pipeline"""
        # Load image based on input type
        if isinstance(image_path, str):
            # File path - use cv2 for proper color handling
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            # cv2 loads in BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # PIL Image from Gradio - convert directly to numpy RGB
            img = np.array(image_path)
            
            # Handle different image formats
            if img.ndim == 3 and img.shape[2] == 4:
                # RGBA image
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            elif img.ndim == 3 and img.shape[2] == 3:
                # Already RGB from PIL
                pass
            elif img.ndim == 2:
                # Grayscale - convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                raise ValueError(f"Unexpected image format: shape={img.shape}")
        
        # Ensure proper data type and range
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Store original dimensions for later use
        self.original_shape = img.shape
        
        # Convert to tensor in [0, 1] range (required for neural networks)
        img_float = img.astype(np.float32) / 255.0
        
        # Convert to tensor (1, C, H, W) format for batch processing
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor, img
    
    def run_yolo_inference(self, model, img_tensor, conf_threshold=0.25):
        """Run YOLO inference and return detections - using proven ultralytics pipeline"""
        try:
            with torch.no_grad():
                # Store original device and batch info
                original_device = img_tensor.device
                is_batched = img_tensor.shape[0] == 1
                
                # Convert tensor to numpy format for ultralytics
                # Input tensor is (1, C, H, W) with values in [0, 1]
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Scale to [0, 255] range
                img_np = (img_np * 255.0).astype(np.uint8)
                
                # Ensure image is contiguous in memory
                img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
                
                # Run YOLO inference directly on numpy image
                # ultralytics handles all preprocessing internally
                results = model(img_np, verbose=False, conf=conf_threshold)
                
                # Extract detections from results
                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy  # (N, 4) in xyxy format
                    scores = results[0].boxes.conf  # (N,)
                    classes = results[0].boxes.cls   # (N,)
                    
                    if len(boxes) > 0:
                        # Convert to tensors and move to original device
                        boxes = torch.from_numpy(boxes).float().to(original_device)
                        scores = torch.from_numpy(scores).float().to(original_device)
                        classes = torch.from_numpy(classes).float().to(original_device)
                        
                        # Format: [x1, y1, x2, y2, conf, class_id]
                        detections = torch.cat([
                            boxes,
                            scores.unsqueeze(1),
                            classes.unsqueeze(1)
                        ], dim=1)
                    else:
                        detections = torch.empty((0, 6), device=original_device)
                else:
                    detections = torch.empty((0, 6), device=original_device)
            
            return detections
        except Exception as e:
            print(f"Error in YOLO inference: {e}")
            import traceback
            traceback.print_exc()
            return torch.empty((0, 6))
    
    def run_faster_rcnn_inference(self, model, img_tensor, conf_threshold=0.3):
        """Run Faster R-CNN inference and return detections"""
        try:
            with torch.no_grad():
                predictions = model([img_tensor.squeeze(0)])
                
                # Process predictions
                boxes = predictions[0]['boxes']
                scores = predictions[0]['scores']
                labels = predictions[0]['labels']
                
                # Filter by confidence
                keep = scores >= conf_threshold
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Format as [x1, y1, x2, y2, conf, class_id]
                if len(boxes) > 0:
                    detections = torch.cat([
                        boxes,
                        scores.unsqueeze(1),
                        labels.float().unsqueeze(1)
                    ], dim=1)
                else:
                    detections = torch.empty((0, 6))
            
            return detections
        except Exception as e:
            print(f"Error in Faster R-CNN inference: {e}")
            return torch.empty((0, 6))
    
    def run_inference(self, model_name, img_tensor, conf_threshold=None):
        """Run inference with the specified model"""
        model, _ = self.model_loader.get_current_model()
        
        # Ensure model is in eval mode for consistent inference
        if hasattr(model, 'eval'):
            model.eval()
        
        if model_name == 'YOLOv5':
            # Use optimized confidence threshold for YOLO (0.35 for better accuracy)
            threshold = conf_threshold if conf_threshold is not None else 0.35
            return self.run_yolo_inference(model, img_tensor, threshold)
        elif model_name == 'Faster R-CNN':
            # Faster R-CNN uses default 0.3
            threshold = conf_threshold if conf_threshold is not None else 0.3
            return self.run_faster_rcnn_inference(model, img_tensor, threshold)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def calculate_metrics(self, original_dets, adversarial_dets):
        """Calculate comparison metrics"""
        metrics = {
            'original_count': len(original_dets),
            'adversarial_count': len(adversarial_dets),
            'original_avg_conf': 0.0,
            'adversarial_avg_conf': 0.0,
            'attack_success': False,
            'class_changes': []
        }
        
        # Calculate average confidence
        if len(original_dets) > 0:
            metrics['original_avg_conf'] = torch.mean(original_dets[:, 4]).item()
        
        if len(adversarial_dets) > 0:
            metrics['adversarial_avg_conf'] = torch.mean(adversarial_dets[:, 4]).item()
        
        # Determine attack success
        if len(original_dets) > len(adversarial_dets):
            metrics['attack_success'] = True
        elif len(original_dets) == len(adversarial_dets):
            # Check if confidence decreased
            if metrics['adversarial_avg_conf'] < metrics['original_avg_conf']:
                metrics['attack_success'] = True
        
        # Class changes (simplified)
        if len(original_dets) > 0 and len(adversarial_dets) > 0:
            orig_classes = set(original_dets[:, 5].int().tolist())
            adv_classes = set(adversarial_dets[:, 5].int().tolist())
            metrics['class_changes'] = list(orig_classes.symmetric_difference(adv_classes))
        
        return metrics
    
    def tensor_to_numpy_image(self, img_tensor):
        """Convert tensor [0,1] to numpy uint8 image"""
        # Handle batch dimension
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
        
        # Ensure proper shape (C, H, W)
        if img_tensor.shape[0] in [1, 3, 4]:
            img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
        else:
            img_np = img_tensor.detach().cpu().numpy()
        
        # Scale to [0, 255]
        img_np = np.clip(img_np * 255.0, 0, 255).astype('uint8')
        
        # Handle grayscale
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=2)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        
        return img_np
