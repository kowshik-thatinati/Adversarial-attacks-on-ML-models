import os
import argparse
import torch
import numpy as np
from models.faster_rcnn import FasterRCNNModel
from attacks import FGSMAttack
from visualize import plot_comparison
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run Faster R-CNN with adversarial attack')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--epsilon', type=float, default=0.20, help='Attack strength (0-1)')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold (0-1)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading Faster R-CNN model...")
    model = FasterRCNNModel(device=device)
    attack = FGSMAttack(model.model, model_name='faster_rcnn', epsilon=args.epsilon)

    print(f"Processing {args.image}...")
    detections, original_image, img_tensor = model.process_image(args.image, conf_threshold=args.conf)

    if len(detections) == 0:
        print("No objects detected in the original image. Exiting.")
        return

    print(f"Found {len(detections)} objects in original image")

    coco_classes = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench']
    print("\nOriginal detections:")
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        class_name = coco_classes[int(cls_id)] if int(cls_id) < len(coco_classes) else f"Class {int(cls_id)}"
        print(f"  {class_name}: {conf:.3f}")

    print("\nGenerating adversarial example (running real model gradients)...")
    adv_img = attack.attack(img_tensor, detections)

    # Get REAL predictions on the adversarial image — no post-hoc editing.
    with torch.no_grad():
        adv_predictions = model.predict(adv_img)
        boxes = adv_predictions[0]['boxes']
        scores = adv_predictions[0]['scores']
        labels = adv_predictions[0]['labels']

        keep = scores >= args.conf
        adv_detections = torch.cat([
            boxes[keep],
            scores[keep].unsqueeze(1),
            labels[keep].float().unsqueeze(1)
        ], dim=1)

    print(f"\nAdversarial detections (actual model output on perturbed image):")
    if len(adv_detections) == 0:
        print("  (no detections above confidence threshold)")
    else:
        for det in adv_detections:
            x1, y1, x2, y2, conf, cls_id = det[:6]
            class_name = coco_classes[int(cls_id)] if int(cls_id) < len(coco_classes) else f"Class {int(cls_id)}"
            print(f"  {class_name}: {conf:.3f}")

    print(f"\nFound {len(detections)} objects before attack, {len(adv_detections)} after")
    if len(detections) > 0:
        orig_mean_conf = detections[:, 4].mean().item()
        print(f"Mean confidence before: {orig_mean_conf:.3f}")
    if len(adv_detections) > 0:
        adv_mean_conf = adv_detections[:, 4].mean().item()
        print(f"Mean confidence after:  {adv_mean_conf:.3f}")

    adv_image_np = adv_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    adv_image_np = np.clip(adv_image_np * 255.0, 0, 255).astype('uint8')

    output_path = os.path.join(args.output, f"faster_rcnn_attack_{Path(args.image).name}")
    fig = plot_comparison(
        original_image, detections.cpu().numpy(),
        adv_image_np, adv_detections.cpu().numpy(),
        class_names=model.classes,
        model_name="Faster R-CNN"
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()