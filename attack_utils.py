import torch
import numpy as np
from attacks.utils import iou


class FGSMAttack:
    """
    Iterative FGSM attack against a real object detector.

    Unlike the previous implementation, every iteration performs an actual
    forward pass through the target model and backpropagates through it.
    The perturbation direction is therefore driven by the model's own
    response to the image, not by a hand-crafted "push pixels toward grey"
    heuristic.
    """

    def __init__(self, model, model_name, epsilon=0.05, attack_mode='entire_image', num_iterations=10, **kwargs):
        """
        Args:
            model: the underlying torch model
                   - for Faster R-CNN: the torchvision detection model directly
                   - for YOLO: the ultralytics `YOLO` wrapper (we reach into
                     `.model` internally to get the raw torch nn.Module)
            model_name: 'yolo' or 'faster_rcnn'
            epsilon: total perturbation budget (L-infinity, in [0,1] pixel space)
            attack_mode: UNUSED, kept only so old positional call sites (like
                         app.py, which calls FGSMAttack(model, model_name,
                         epsilon, 'entire_image') positionally) don't
                         accidentally land a string into num_iterations.
                         Every attack is now real/model-gradient-based
                         regardless of this value.
            num_iterations: number of I-FGSM steps. Pass this as a KEYWORD
                             argument (num_iterations=...) from any new call
                             site, never positionally, to avoid this exact bug.
            **kwargs: accepted and ignored for any further backward compatibility.
        """
        self.model = model
        self.model_name = model_name
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        if kwargs:
            print(f"Note: ignoring unused FGSMAttack kwargs: {list(kwargs.keys())}")

    def attack(self, image_tensor, detections):
        """
        Generate an adversarial image via iterative FGSM.

        Args:
            image_tensor: (1, C, H, W) tensor in [0, 1]
            detections: (N, 6) tensor [x1, y1, x2, y2, conf, class_id] from the
                        CLEAN image — used as the reference to attack against
        Returns:
            Adversarial image tensor, same shape, clamped to [0, 1]
        """
        if len(detections) == 0:
            return image_tensor

        # image_tensor may have been created upstream inside a
        # torch.inference_mode() context (e.g. during the original-image
        # detection pass). Tensors created that way are permanently flagged
        # as "inference tensors" and can never be used with autograd again,
        # even after .detach()/.clone() — UNLESS the clone itself happens
        # outside an active inference_mode context. This forces that.
        with torch.inference_mode(False):
            adv_image = image_tensor.detach().clone()

        step_epsilon = self.epsilon / self.num_iterations

        for iteration in range(self.num_iterations):
            adv_image = adv_image.detach().requires_grad_(True)

            loss = self._compute_loss(adv_image, detections)

            if loss is None:
                print(f"Warning: could not compute a loss at iteration {iteration}, "
                      f"stopping early (kept {iteration} of {self.num_iterations} steps)")
                break

            if hasattr(self.model, 'zero_grad'):
                self.model.zero_grad(set_to_none=True)

            loss.backward()

            if adv_image.grad is None:
                print(f"Warning: no gradient reached the input at iteration {iteration}, "
                      f"stopping early")
                break

            with torch.no_grad():
                # loss is constructed (see _compute_loss) so that INCREASING it
                # corresponds to DEGRADING detection quality. Gradient ascent
                # (+ sign) therefore pushes the image toward fooling the model.
                adv_image = adv_image.detach() + step_epsilon * adv_image.grad.sign()
                adv_image = torch.clamp(adv_image, 0.0, 1.0)

        return adv_image.detach()

    def _compute_loss(self, adv_image, detections):
        if self.model_name == 'faster_rcnn':
            return self._faster_rcnn_loss(adv_image, detections)
        else:
            return self._yolo_loss(adv_image, detections)

    def _faster_rcnn_loss(self, adv_image, detections):
        """
        Uses the model's own training-mode detection loss (classification +
        box regression), computed against the CLEAN-image detections treated
        as ground truth. Maximizing this loss via gradient ascent pushes the
        model away from correctly detecting what it originally detected —
        a standard formulation for untargeted adversarial attacks on
        detectors (see e.g. Xie et al., "Adversarial Examples for Semantic
        Segmentation and Object Detection", 2017).
        """
        boxes = detections[:, :4].detach()
        labels = detections[:, 5].detach().long().clamp(min=1)

        if boxes.shape[0] == 0:
            return None

        targets = [{'boxes': boxes, 'labels': labels}]

        was_training = self.model.training
        self.model.train()
        try:
            loss_dict = self.model([adv_image.squeeze(0)], targets)
            loss = sum(v for v in loss_dict.values())
        except Exception as e:
            print(f"Error computing Faster R-CNN loss: {e}")
            loss = None
        finally:
            if not was_training:
                self.model.eval()

        return loss

    def _yolo_loss(self, adv_image, detections):
        """
        Runs the RAW underlying torch model directly (self.model.model),
        bypassing ultralytics' high-level predict pipeline — that pipeline
        converts tensors to numpy internally and detaches gradients, which
        silently breaks any attack built on top of it.

        Raw output formats vary across ultralytics checkpoint types (classic
        anchor-based YOLOv5 vs. the unified/anchor-free 'u' variants like
        yolov5su.pt), and even across ultralytics library versions. This
        method handles several known shapes defensively and prints a
        diagnostic on first failure instead of crashing, so you can see
        exactly what format your specific model/version returns.
        """
        underlying = self.model.model if hasattr(self.model, 'model') else self.model
        underlying.eval()  # eval mode, NOT train — matches the verified-working
                            # notebook pattern; train mode changes the Detect
                            # head's output shape and caused the dict output seen before

        try:
            with torch.enable_grad():
                raw_output = underlying(adv_image)
        except Exception as e:
            print(f"[FGSMAttack/_yolo_loss] Error during raw forward pass: {e}")
            return None

        raw_preds = self._extract_prediction_tensor(raw_output)

        if raw_preds is None:
            print("[FGSMAttack/_yolo_loss] Could not extract a usable prediction "
                  "tensor from the raw model output. Skipping this attack step. "
                  f"Raw output type was: {type(raw_output)}. "
                  + (f"Dict keys: {list(raw_output.keys())}" if isinstance(raw_output, dict) else ""))
            return None

        if raw_preds.dim() == 3:
            raw_preds = raw_preds[0]  # drop batch dim -> (num_preds, 5+num_classes)
        elif raw_preds.dim() != 2:
            print(f"[FGSMAttack/_yolo_loss] Unexpected prediction tensor shape "
                  f"{tuple(raw_preds.shape)} (expected 2D or 3D). Skipping this step.")
            return None

        if raw_preds.shape[-1] < 5:
            print(f"[FGSMAttack/_yolo_loss] Prediction tensor's last dimension is "
                  f"{raw_preds.shape[-1]}, too small to contain a confidence score "
                  f"at index 4. This model's output format doesn't match the "
                  f"assumed [x,y,w,h,conf,...classes] layout. Skipping this step.")
            return None

        boxes_raw = raw_preds[:, :4].detach().cpu().numpy()

        matched_confidences = []
        for det in detections:
            x1, y1, x2, y2 = det[:4].detach().cpu().numpy()
            ious = [iou([x1, y1, x2, y2], box) for box in boxes_raw]
            if len(ious) == 0:
                continue
            best_idx = int(np.argmax(ious))
            if ious[best_idx] <= 0:
                continue
            matched_confidences.append(raw_preds[best_idx, 4])

        if not matched_confidences:
            print("[FGSMAttack/_yolo_loss] No original detections could be matched "
                  "to a raw prediction box (zero IoU overlap for all). Skipping this step.")
            return None

        total_confidence = torch.stack(matched_confidences).sum()
        return -total_confidence

    @staticmethod
    def _extract_prediction_tensor(raw_output):
        """
        Best-effort extraction of a prediction tensor from whatever the raw
        model call returned. Handles: plain tensor, list/tuple (takes first
        tensor element found), and dict (tries common key names, otherwise
        gives up and returns None so the caller can log diagnostics).
        """
        if torch.is_tensor(raw_output):
            return raw_output

        if isinstance(raw_output, (list, tuple)):
            for item in raw_output:
                if torch.is_tensor(item):
                    return item
                if isinstance(item, (list, tuple)) and len(item) > 0 and torch.is_tensor(item[0]):
                    return item[0]
            return None

        if isinstance(raw_output, dict):
            for key in ('one2one', 'one2many', 'pred', 'preds', 'predictions', 'det', 'output'):
                if key in raw_output and torch.is_tensor(raw_output[key]):
                    return raw_output[key]
            for value in raw_output.values():
                if torch.is_tensor(value):
                    return value
            return None

        return None