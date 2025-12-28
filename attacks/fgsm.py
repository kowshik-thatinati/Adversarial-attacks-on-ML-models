import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import iou

class FGSMAttack:
    def __init__(self, model, epsilon=0.05):
        """
        Initialize FGSM attack
        
        Args:
            model: The target model (YOLO or Faster R-CNN)
            epsilon: Perturbation magnitude (0-1)
        """
        self.model = model
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()
        
    def attack(self, image_tensor, detections, target_class=None):
        """
        Generate adversarial example using FGSM
        
        Args:
            image_tensor: Input image tensor (1, C, H, W)
            detections: Original detections [x1, y1, x2, y2, conf, class_id]
            target_class: Target class for misclassification (None for untargeted)
            
        Returns:
            Adversarial image tensor
        """
        adv_image = image_tensor.clone().detach().requires_grad_(True)

        mask = torch.zeros_like(adv_image)
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = map(int, (x1.item(), y1.item(), x2.item(), y2.item()))
            x1 = max(0, min(x1, adv_image.shape[-1] - 1))
            x2 = max(0, min(x2, adv_image.shape[-1]))
            y1 = max(0, min(y1, adv_image.shape[-2] - 1))
            y2 = max(0, min(y2, adv_image.shape[-2]))
            if x2 > x1 and y2 > y1:
                mask[:, :, y1:y2, x1:x2] = 1.0

        loss = None
        if hasattr(self.model, 'non_max_suppression'):
            output = self.model.model(adv_image)
            pred_boxes = self.model.non_max_suppression(output, conf_thres=0.25)[0]
            if len(pred_boxes) > 0 and len(detections) > 0:
                det0 = detections[0]
                ious = [iou(det0[:4].detach().cpu().numpy().flatten(), box[:4].cpu().numpy())
                        for box in pred_boxes.boxes.xyxy]
                if ious:
                    max_idx = int(np.argmax(ious))
                    pred_conf = pred_boxes.boxes.conf[max_idx]
                    if pred_conf.requires_grad:
                        loss = -pred_conf
        else:
            if len(detections) > 0:
                boxes = detections[:, :4].detach()
                labels = detections[:, 5].detach().long().clamp(min=1)
                targets = [{
                    'boxes': boxes,
                    'labels': labels,
                }]

                was_training = self.model.model.training
                self.model.model.train()
                model_in_list = [adv_image.squeeze(0)]
                losses = self.model.model(model_in_list, targets)
                loss = sum(v for v in losses.values())
                if not was_training:
                    self.model.model.eval()

        if loss is None:
            return image_tensor

        loss.backward()
        perturb = self.epsilon * adv_image.grad.sign()
        adv_image = adv_image.detach() + mask * perturb
        adv_image = torch.clamp(adv_image, 0.0, 1.0)

        return adv_image.detach()
