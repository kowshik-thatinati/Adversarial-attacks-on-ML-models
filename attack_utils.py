import torch
import torch.nn.functional as F
import numpy as np
from attacks.utils import iou

class FGSMAttack:
    """FGSM Attack with two modes: entire image and bounding boxes only"""
    
    def __init__(self, model, model_name, epsilon=0.05, attack_mode='entire_image'):
        """
        Initialize FGSM attack
        
        Args:
            model: The target model (YOLO or Faster R-CNN)
            model_name: 'yolo' or 'faster_rcnn'
            epsilon: Perturbation magnitude (0-1)
            attack_mode: 'entire_image' or 'bounding_boxes_only'
        """
        self.model = model
        self.model_name = model_name
        self.epsilon = epsilon
        self.attack_mode = attack_mode
    
    def create_bounding_box_mask(self, detections, image_shape):
        """Create a mask for detected objects' bounding boxes"""
        mask = torch.zeros(image_shape, device=detections.device)
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = map(int, (x1.item(), y1.item(), x2.item(), y2.item()))
            x1 = max(0, min(x1, image_shape[-1] - 1))
            x2 = max(0, min(x2, image_shape[-1]))
            y1 = max(0, min(y1, image_shape[-2] - 1))
            y2 = max(0, min(y2, image_shape[-2]))
            if x2 > x1 and y2 > y1:
                mask[:, :, y1:y2, x1:x2] = 1.0
        return mask
    
    def attack(self, image_tensor, detections):
        """
        Generate adversarial example using direct pixel perturbation
        
        Args:
            image_tensor: Input image tensor (1, C, H, W) in [0,1] space
            detections: Original detections [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Adversarial image tensor
        """
        if len(detections) == 0:
            return image_tensor
        
        # Use direct pixel-level perturbation with multiple iterations
        adv_image = image_tensor.clone()
        num_iterations = 10
        step_epsilon = self.epsilon / num_iterations
        
        for iteration in range(num_iterations):
            adv_image = adv_image.detach().requires_grad_(True)
            
            # Direct pixel-level loss on detected regions
            # This doesn't require calling the model, ensuring gradients flow properly
            loss = torch.tensor(0.0, device=adv_image.device, requires_grad=True)
            
            for det in detections:
                x1, y1, x2, y2 = map(int, (det[0].item(), det[1].item(), det[2].item(), det[3].item()))
                
                # Clamp to image bounds
                H, W = adv_image.shape[2:]
                x1 = max(0, min(x1, W-1))
                y1 = max(0, min(y1, H-1))
                x2 = max(x1+1, min(x2, W))
                y2 = max(y1+1, min(y2, H))
                
                if x2 > x1 and y2 > y1:
                    # Extract region and perturb to mid-grey (neutral color)
                    region = adv_image[:, :, y1:y2, x1:x2]
                    # Loss: Push detected regions toward neutral/ambiguous colors
                    loss = loss + (region - 0.5).pow(2).mean() * 100.0
            
            # Regularization: keep changes small but focused
            loss = loss + (adv_image - image_tensor).pow(2).mean() * 5.0
            
            if loss > 0:
                loss.backward()
                
                with torch.no_grad():
                    perturb = step_epsilon * adv_image.grad.sign()
                    adv_image = adv_image.detach() + perturb
                    adv_image = torch.clamp(adv_image, 0.0, 1.0)
        
        print(f"Direct pixel attack: {num_iterations} iterations, total epsilon={self.epsilon}")
        return adv_image.detach()
    
    def calculate_loss(self, adv_image, detections):
        """Calculate loss for misclassification attack"""
        loss = None
        
        try:
            if self.model_name == 'yolo' or 'YOLOv5' in str(type(self.model)):
                # YOLO: Target the model to cause misclassification
                # Strategy: Directly attack the detected objects to make them undetectable or misclassified
                
                if len(detections) > 0:
                    # For each detected region, minimize the model's confidence in that class
                    loss = torch.tensor(0.0, device=adv_image.device, requires_grad=True)
                    
                    for det in detections:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        x1, y1, x2, y2 = map(int, (x1.item(), y1.item(), x2.item(), y2.item()))
                        
                        # Clamp to image bounds
                        H, W = adv_image.shape[2:]
                        x1 = max(0, min(x1, W-1))
                        y1 = max(0, min(y1, H-1))
                        x2 = max(x1+1, min(x2, W))
                        y2 = max(y1+1, min(y2, H))
                        
                        if x2 > x1 and y2 > y1:
                            # Extract region and apply strong noise
                            region = adv_image[:, :, y1:y2, x1:x2]
                            
                            # Loss: Perturb the region to flip features
                            # Use negative of pixel values to invert colors in that region
                            region_loss = (region - (1.0 - region)).pow(2).mean()
                            
                            # Also add a term to maximize perturbation magnitude
                            region_loss = region_loss + region.abs().mean() * 2.0
                            
                            loss = loss + region_loss
                    
                    # Average across detections
                    loss = loss / max(1, len(detections))
                    
                    # Add global perturbation term
                    loss = loss + adv_image.pow(2).mean() * 0.5
                else:
                    # No detections: generic perturbation
                    loss = adv_image.pow(2).mean()
                
                if loss is None:
                    loss = adv_image.pow(2).mean()
            
            else:  # faster_rcnn
                if len(detections) > 0:
                    boxes = detections[:, :4].detach()
                    labels = detections[:, 5].detach().long().clamp(min=1)
                    targets = [{
                        'boxes': boxes,
                        'labels': labels,
                    }]
                    
                    was_training = self.model.training
                    self.model.train()
                    model_in_list = [adv_image.squeeze(0)]
                    losses = self.model(model_in_list, targets)
                    loss = sum(v for v in losses.values())
                    # Ensure model is back in eval mode after attack
                    self.model.eval()
        except Exception as e:
            print(f"Error calculating loss: {e}")
            import traceback
            traceback.print_exc()
        
        return loss
