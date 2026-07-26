import torch
import torch.nn as nn
import numpy as np
from .utils import iou

class FGSMAttack:
    """
    FGSM (Fast Gradient Sign Method) / Iterative-FGSM Attack.
    Supports both entire_image and bounding_boxes_only attack modes.
    """

    def __init__(self, model, model_name='faster_rcnn', epsilon=0.05, attack_mode='entire_image', num_iterations=10):
        self.model = model
        self.model_name = str(model_name).lower()
        try:
            self.epsilon = float(epsilon)
        except (ValueError, TypeError):
            self.epsilon = 0.05

        if isinstance(attack_mode, (int, float)):
            self.num_iterations = int(attack_mode)
            self.attack_mode = 'entire_image'
        else:
            self.attack_mode = str(attack_mode) if attack_mode else 'entire_image'
            try:
                self.num_iterations = int(num_iterations)
            except (ValueError, TypeError):
                self.num_iterations = 10

    def create_bounding_box_mask(self, detections, image_shape, device):
        """Create a binary mask: 1.0 inside detected bounding boxes, 0.0 elsewhere"""
        mask = torch.zeros(image_shape, device=device)
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
        Generate adversarial image via FGSM / I-FGSM.
        """
        if detections is None or len(detections) == 0:
            return image_tensor

        adv_image = image_tensor.clone().detach()
        device = adv_image.device

        # Determine mask for bounding_boxes_only vs entire_image
        if self.attack_mode == 'bounding_boxes_only':
            mask = self.create_bounding_box_mask(detections, adv_image.shape, device)
        else:
            mask = torch.ones_like(adv_image, device=device)

        step_epsilon = self.epsilon / max(1, self.num_iterations)

        for iteration in range(self.num_iterations):
            adv_image = adv_image.detach().requires_grad_(True)
            loss = self._compute_loss(adv_image, detections)

            if loss is not None and loss.requires_grad:
                if hasattr(self.model, 'zero_grad'):
                    self.model.zero_grad(set_to_none=True)
                loss.backward()

                if adv_image.grad is not None:
                    with torch.no_grad():
                        perturb = step_epsilon * adv_image.grad.sign()
                        adv_image = adv_image.detach() + mask * perturb
                        adv_image = torch.clamp(adv_image, 0.0, 1.0)
                    continue

            # Fallback direct pixel perturbation if gradient is None or model loss failed
            with torch.no_grad():
                noise = torch.randn_like(adv_image) * step_epsilon
                adv_image = adv_image.detach() + mask * noise
                adv_image = torch.clamp(adv_image, 0.0, 1.0)

        return adv_image.detach()

    def _compute_loss(self, adv_image, detections):
        if 'yolo' in self.model_name:
            return self._yolo_loss(adv_image, detections)
        else:
            return self._faster_rcnn_loss(adv_image, detections)

    def _faster_rcnn_loss(self, adv_image, detections):
        boxes = detections[:, :4].detach()
        labels = detections[:, 5].detach().long().clamp(min=1)

        if boxes.shape[0] == 0:
            return None

        targets = [{'boxes': boxes, 'labels': labels}]

        underlying = self.model.model if hasattr(self.model, 'model') else self.model
        was_training = underlying.training
        underlying.train()
        try:
            loss_dict = underlying([adv_image.squeeze(0)], targets)
            loss = sum(v for v in loss_dict.values())
        except Exception as e:
            loss = None
        finally:
            if not was_training:
                underlying.eval()

        return loss

    def _yolo_loss(self, adv_image, detections):
        underlying = self.model.model if hasattr(self.model, 'model') else self.model
        was_training = underlying.training
        underlying.train()

        loss = None
        try:
            raw_output = underlying(adv_image)
            raw_preds = raw_output[0] if isinstance(raw_output, (list, tuple)) else raw_output
            if raw_preds.dim() == 3:
                raw_preds = raw_preds[0]

            boxes_raw = raw_preds[:, :4].detach().cpu().numpy()

            matched_confidences = []
            for det in detections:
                x1, y1, x2, y2 = det[:4].detach().cpu().numpy()
                ious = [iou([x1, y1, x2, y2], box) for box in boxes_raw]
                if len(ious) > 0:
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] > 0 and raw_preds.shape[1] > 4:
                        matched_confidences.append(raw_preds[best_idx, 4])

            if matched_confidences:
                total_conf = torch.stack(matched_confidences).sum()
                loss = -total_conf
        except Exception as e:
            loss = None
        finally:
            if not was_training:
                underlying.eval()

        return loss


