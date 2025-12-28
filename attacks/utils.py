import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

def iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        boxA: [x1, y1, x2, y2]
        boxB: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Ensure boxes are 1D arrays
    boxA = np.asarray(boxA).flatten()
    boxB = np.asarray(boxB).flatten()
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-6)

def match_detections(dets_before: torch.Tensor, 
                    dets_after: torch.Tensor, 
                    iou_threshold: float = 0.5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Match detections before and after attack
    
    Args:
        dets_before: Detections before attack [N, 6] (x1, y1, x2, y2, conf, class_id)
        dets_after: Detections after attack [M, 6]
        iou_threshold: Minimum IoU to consider a match
        
    Returns:
        List of matched detections (det_before, det_after)
    """
    matched = []
    used_after_idxs = set()
    
    for i, det_b in enumerate(dets_before):
        best_iou = 0
        best_j = None
        
        for j, det_a in enumerate(dets_after):
            if j in used_after_idxs:
                continue
                
            iou_score = iou(det_b[:4].cpu().numpy(), det_a[:4].cpu().numpy())
            
            if iou_score > best_iou:
                best_iou = iou_score
                best_j = j
                
        if best_iou >= iou_threshold:
            matched.append((det_b, dets_after[best_j]))
            used_after_idxs.add(best_j)
        else:
            matched.append((det_b, None))
            
    # Add remaining detections that weren't matched
    for j in range(len(dets_after)):
        if j not in used_after_idxs:
            matched.append((None, dets_after[j]))
            
    return matched

def plot_detections(image: np.ndarray, 
                   detections: torch.Tensor, 
                   class_names: list = None,
                   title: str = "Detections") -> None:
    """
    Plot image with detections
    
    Args:
        image: Input image (H, W, 3) in RGB format
        detections: Detections [N, 6] (x1, y1, x2, y2, conf, class_id)
        class_names: List of class names
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(cls_id)
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_names[cls_id]} {conf:.2f}" if class_names else f"{cls_id} {conf:.2f}"
        plt.text(
            x1, y1 - 5, label,
            color='yellow', fontsize=12, weight='bold',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()  # Return the figure object
