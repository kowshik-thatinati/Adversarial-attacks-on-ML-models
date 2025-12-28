import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_detections(image, detections, title, class_names=None):
    """
    Plot image with detections - boxes clipped to image boundaries
    
    Args:
        image: Input image (H, W, 3) in RGB format
        detections: Detections [N, 6] (x1, y1, x2, y2, conf, class_id)
        title: Plot title
        class_names: List of class names
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(cls_id)
        
        # CLIP boxes to image boundaries
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip if box is invalid after clipping
        if x2 <= x1 or y2 <= y1:
            continue
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        if class_names is not None and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label = f"{cls_id} {conf:.2f}"
        
        # Position label inside image bounds
        label_y = max(10, y1 - 5)
        plt.text(
            x1, label_y, label,
            color='yellow', fontsize=10, weight='bold',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.title(title, fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def plot_comparison(original_image, original_dets, 
                   adv_image, adv_dets, 
                   class_names=None, model_name="YOLO"):
    """
    Plot comparison between original and adversarial detections - boxes clipped to image boundaries
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get image dimensions
    orig_height, orig_width = original_image.shape[:2]
    adv_height, adv_width = adv_image.shape[:2]
    
    # Plot original detections
    ax1.imshow(original_image)
    ax1.set_title(f"Original Detections ({model_name})", fontsize=14, weight='bold')
    
    for det in original_dets:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(cls_id)
        
        # CLIP boxes to image boundaries
        x1 = max(0, min(x1, orig_width - 1))
        y1 = max(0, min(y1, orig_height - 1))
        x2 = max(0, min(x2, orig_width))
        y2 = max(0, min(y2, orig_height))
        
        # Skip if box is invalid
        if x2 <= x1 or y2 <= y1:
            continue
        
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='lime', facecolor='none'
        )
        ax1.add_patch(rect)
        
        if class_names is not None and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label = f"{cls_id} {conf:.2f}"
        
        label_y = max(10, y1 - 5)
        ax1.text(
            x1, label_y, label,
            color='yellow', fontsize=10, weight='bold',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    ax1.axis('off')
    
    # Plot adversarial detections
    ax2.imshow(adv_image)
    ax2.set_title(f"Adversarial Detections ({model_name})", fontsize=14, weight='bold')
    
    for det in adv_dets:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cls_id = int(cls_id)
        
        # CLIP boxes to image boundaries
        x1 = max(0, min(x1, adv_width - 1))
        y1 = max(0, min(y1, adv_height - 1))
        x2 = max(0, min(x2, adv_width))
        y2 = max(0, min(y2, adv_height))
        
        # Skip if box is invalid
        if x2 <= x1 or y2 <= y1:
            continue
        
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax2.add_patch(rect)
        
        if class_names is not None and cls_id < len(class_names):
            label = f"{class_names[cls_id]} {conf:.2f}"
        else:
            label = f"{cls_id} {conf:.2f}"
        
        label_y = max(10, y1 - 5)
        ax2.text(
            x1, label_y, label,
            color='yellow', fontsize=10, weight='bold',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    ax2.axis('off')
    plt.tight_layout()
    return fig
