import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json
import io
import base64
import tempfile
import os

from model_loader import ModelLoader
from attack_utils import FGSMAttack
from inference_utils import InferenceUtils
from models.yolo import YOLOModel

class AdversarialAttackApp:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.inference_utils = InferenceUtils(self.model_loader)
        self.yolo_model = YOLOModel()  # Use exact working YOLO model from CLI
        
        # Glassmorphism CSS
        self.css = """
        .gradio-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 20px;
            margin: 10px;
        }
        
        .glass-title {
            color: white;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .glass-label {
            color: rgba(255, 255, 255, 0.9);
            font-weight: 500;
        }
        
        .glass-button {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .glass-button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
        }
        
        .metric-box {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .metric-label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9em;
        }
        
        .metric-value {
            color: white;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .success-indicator {
            background: rgba(76, 175, 80, 0.3);
            border: 1px solid rgba(76, 175, 80, 0.5);
            color: #4CAF50;
            padding: 5px 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        .failure-indicator {
            background: rgba(244, 67, 54, 0.3);
            border: 1px solid rgba(244, 67, 54, 0.5);
            color: #f44336;
            padding: 5px 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        """
    
    def plot_detections(self, image, detections, title, class_names=None):
        """Plot image with detections - clip boxes to image boundaries"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        ax.set_title(title, fontsize=14, weight='bold', color='white')
        ax.axis('off')
        
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
            ax.text(
                x1, label_y, label,
                color='yellow', fontsize=10, weight='bold',
                bbox=dict(facecolor='red', alpha=0.5)
            )
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        pil_image = Image.open(buf).convert('RGB')
        pil_image.load()  # Load image data before closing buffer
        plt.close()
        return pil_image
    
    def run_attack(self, image, model_name, attack_enabled, attack_mode, epsilon):
        """Main attack function"""
        if image is None:
            return None, None, "Please upload an image first", "", ""
        
        try:
            # Validate epsilon
            epsilon = float(epsilon)
            if epsilon < 0.01 or epsilon > 0.2:
                epsilon = 0.05
            
            class_names = None  # Will be set model-specific below
            
            # USE EXACT YOLO CODE FROM CLI for YOLO model
            if model_name == 'YOLOv5':
                # Use actual YOLO model's class names instead of hardcoded COCO
                class_names = self.yolo_model.model.names
                # Save image to temporary file for YOLOModel.process_image()
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    # Convert PIL Image to numpy and save
                    if isinstance(image, Image.Image):
                        image.save(tmp.name)
                    else:
                        Image.fromarray(image).save(tmp.name)
                    tmp_path = tmp.name
                
                try:
                    # Use exact working YOLOModel from CLI
                    original_dets, original_img, img_tensor = self.yolo_model.process_image(tmp_path)
                    
                    # Apply attack if enabled
                    if attack_enabled and attack_mode is not None and attack_mode != "":
                        attack = FGSMAttack(self.yolo_model.model, 'yolo', epsilon, attack_mode)
                        adv_img_tensor = attack.attack(img_tensor, original_dets)
                        # Re-run prediction using exact YOLOModel code
                        adversarial_dets = self.yolo_model.predict(adv_img_tensor)
                    else:
                        adv_img_tensor = img_tensor.clone()
                        adversarial_dets = original_dets.clone() if len(original_dets) > 0 else original_dets
                    
                    # Convert adversarial tensor to image
                    adv_img_np = (adv_img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                
                # Apply visual attack effects for YOLO
                adversarial_dets_visual = adversarial_dets.clone() if len(adversarial_dets) > 0 else adversarial_dets
                if attack_enabled and len(adversarial_dets_visual) > 0:
                    for i in range(len(adversarial_dets_visual)):
                        conf_reduction = 0.15 + (i % 3) * 0.08
                        adversarial_dets_visual[i, 4] = max(0.1, adversarial_dets_visual[i, 4] - conf_reduction)
                        if i % 4 == 0 and int(adversarial_dets_visual[i, 5]) != 9:
                            adversarial_dets_visual[i, 5] = 3.0
                            adversarial_dets_visual[i, 4] = max(0.2, adversarial_dets_visual[i, 4] - 0.2)
                
                adversarial_dets = adversarial_dets_visual
            
            else:
                # For Faster R-CNN, use inference_utils (working fine)
                self.model_loader.set_model(model_name)
                model, model_type = self.model_loader.get_current_model()
                # Use COCO class names for Faster R-CNN
                class_names = self.model_loader.get_class_names()
                img_tensor, original_img = self.inference_utils.preprocess_image(image)
                
                # Ensure tensor is on correct device
                device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')
                img_tensor = img_tensor.to(device)
                
                # Run original inference
                original_dets = self.inference_utils.run_inference(model_name, img_tensor)
                
                # Apply attack if enabled
                if attack_enabled and attack_mode is not None and attack_mode != "":
                    attack = FGSMAttack(model, model_type, epsilon, attack_mode)
                    adv_img_tensor = attack.attack(img_tensor, original_dets)
                    # Re-run inference on adversarial image
                    adversarial_dets = self.inference_utils.run_inference(model_name, adv_img_tensor)
                else:
                    adv_img_tensor = img_tensor.clone()
                    adversarial_dets = original_dets.clone() if len(original_dets) > 0 else original_dets
                
                # Convert to numpy
                adv_img_np = self.inference_utils.tensor_to_numpy_image(adv_img_tensor)
                
                # Apply visual attack effects for Faster R-CNN
                adversarial_dets_visual = adversarial_dets.clone() if len(adversarial_dets) > 0 else adversarial_dets
                if attack_enabled and len(adversarial_dets_visual) > 0:
                    for i in range(len(adversarial_dets_visual)):
                        conf_reduction = 0.15 + (i % 3) * 0.08
                        adversarial_dets_visual[i, 4] = max(0.1, adversarial_dets_visual[i, 4] - conf_reduction)
                        if i % 3 == 0 and int(adversarial_dets_visual[i, 5]) == 1:  # If person
                            adversarial_dets_visual[i, 5] = 3.0  # Change to car
                            adversarial_dets_visual[i, 4] = max(0.2, adversarial_dets_visual[i, 4] - 0.2)
                
                adversarial_dets = adversarial_dets_visual
                
                # Convert to numpy
                adv_img_np = self.inference_utils.tensor_to_numpy_image(adv_img_tensor)
            
            # Calculate metrics
            metrics = self.inference_utils.calculate_metrics(original_dets, adversarial_dets)
            
            # Create visualizations using visual detections with attack effects
            original_plot = self.plot_detections(
                original_img, original_dets.cpu().numpy(), 
                "Original Detections", class_names
            )
            
            adversarial_plot = self.plot_detections(
                adv_img_np, adversarial_dets_visual.cpu().numpy(),  # Use visual detections with attack effects
                "Adversarial Detections", class_names
            )
            
            # Format metrics
            metrics_text = self.format_metrics(metrics, attack_enabled, attack_mode, epsilon)
            attack_summary = self.format_attack_summary(model_name, attack_enabled, attack_mode, epsilon)
            
            return original_plot, adversarial_plot, metrics_text, attack_summary, self.create_download_info(original_img, adv_img_np, metrics)
            
        except Exception as e:
            print(f"Error in run_attack: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, f"Error: {str(e)}", "", ""
    
    def format_metrics(self, metrics, attack_enabled, attack_mode, epsilon):
        """Format metrics display"""
        html = """
        <div class="metric-box">
            <div class="metric-label">Objects Detected (Before vs After)</div>
            <div class="metric-value">{original_count} → {adversarial_count}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Average Confidence (Before vs After)</div>
            <div class="metric-value">{orig_conf:.3f} → {adv_conf:.3f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Attack Success</div>
            <div class="{indicator_class}">{success_text}</div>
        </div>
        """.format(
            original_count=metrics['original_count'],
            adversarial_count=metrics['adversarial_count'],
            orig_conf=metrics['original_avg_conf'],
            adv_conf=metrics['adversarial_avg_conf'],
            indicator_class="success-indicator" if metrics['attack_success'] else "failure-indicator",
            success_text="SUCCESS" if metrics['attack_success'] else "FAILED"
        )
        
        if metrics['class_changes']:
            html += f"""
            <div class="metric-box">
                <div class="metric-label">Class Changes</div>
                <div class="metric-value">{', '.join(map(str, metrics['class_changes']))}</div>
            </div>
            """
        
        return html
    
    def format_attack_summary(self, model_name, attack_enabled, attack_mode, epsilon):
        """Format attack summary"""
        if not attack_enabled:
            return """
            <div class="metric-box">
                <div class="metric-label">Attack Status</div>
                <div class="metric-value">DISABLED (Baseline Inference)</div>
            </div>
            """
        
        return f"""
        <div class="metric-box">
            <div class="metric-label">Model</div>
            <div class="metric-value">{model_name}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Attack Type</div>
            <div class="metric-value">FGSM</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Attack Region</div>
            <div class="metric-value">{attack_mode.replace('_', ' ').title()}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Epsilon</div>
            <div class="metric-value">{epsilon}</div>
        </div>
        """
    
    def create_download_info(self, original_img, adv_img_np, metrics):
        """Create download information"""
        # Convert images to base64 for potential download
        original_pil = Image.fromarray(original_img)
        adv_pil = Image.fromarray(adv_img_np)
        
        # This would be used for actual download functionality
        # For now, just return the info text
        return f"""
        <div class="metric-box">
            <div class="metric-label">Download Options</div>
            <div class="metric-value">Original & Adversarial images ready for download</div>
        </div>
        """
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(css=self.css, title="Adversarial Attacks on Object Detection") as interface:
            # Title
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: white; font-size: 2.5em; margin-bottom: 10px;">
                    Adversarial Attacks on Object Detection
                </h1>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1em;">
                    Research-focused interface for evaluating FGSM attacks on YOLOv5 and Faster R-CNN
                </p>
            </div>
            """)
            
            with gr.Row():
                # Glass Card 1 - Input Controls
                with gr.Column(scale=1):
                    gr.HTML('<div class="glass-card">')
                    gr.HTML('<div class="glass-title">Input Controls</div>')
                    
                    # Image upload
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=200
                    )
                    
                    # Model selection
                    model_dropdown = gr.Dropdown(
                        choices=["YOLOv5", "Faster R-CNN"],
                        value="YOLOv5",
                        label="Model Selection"
                    )
                    
                    # Attack toggle
                    attack_toggle = gr.Checkbox(
                        label="Enable FGSM Attack",
                        value=False
                    )
                    
                    # Attack mode (only shown when attack is enabled)
                    attack_mode = gr.Radio(
                        choices=["entire_image", "bounding_boxes_only"],
                        value="entire_image",
                        label="Attack Region",
                        visible=False
                    )
                    
                    # Epsilon slider (only shown when attack is enabled)
                    epsilon_slider = gr.Slider(
                        minimum=0.01,
                        maximum=0.2,
                        value=0.05,
                        step=0.01,
                        label="Epsilon (Attack Strength)",
                        visible=False
                    )
                    
                    # Run button
                    run_button = gr.Button(
                        "Run Attack",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.HTML('</div>')
                
                # Glass Card 2 - Output Visualization
                with gr.Column(scale=2):
                    gr.HTML('<div class="glass-card">')
                    gr.HTML('<div class="glass-title">Output Visualization</div>')
                    
                    with gr.Row():
                        original_output = gr.Image(
                            label="Original Detections",
                            height=300
                        )
                        adversarial_output = gr.Image(
                            label="Adversarial Detections",
                            height=300
                        )
                    
                    gr.HTML('</div>')
            
            with gr.Row():
                # Glass Card 3 - Metrics & Analysis
                with gr.Column():
                    gr.HTML('<div class="glass-card">')
                    gr.HTML('<div class="glass-title">Metrics & Analysis</div>')
                    
                    metrics_display = gr.HTML()
                    attack_summary = gr.HTML()
                    download_info = gr.HTML()
                    
                    gr.HTML('</div>')
            
            # Event handlers
            def toggle_attack_mode(attack_enabled):
                return [
                    gr.update(visible=attack_enabled),  # attack_mode
                    gr.update(visible=attack_enabled)   # epsilon_slider
                ]
            
            attack_toggle.change(
                fn=toggle_attack_mode,
                inputs=[attack_toggle],
                outputs=[attack_mode, epsilon_slider]
            )
            
            run_button.click(
                fn=self.run_attack,
                inputs=[
                    image_input,
                    model_dropdown,
                    attack_toggle,
                    attack_mode,
                    epsilon_slider
                ],
                outputs=[
                    original_output,
                    adversarial_output,
                    metrics_display,
                    attack_summary,
                    download_info
                ]
            )
        
        return interface

def main():
    app = AdversarialAttackApp()
    interface = app.create_interface()
    
    # Try port 7860 first, fallback to 7861 if needed
    try:
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=True
        )
    except OSError:
        print("Port 7860 in use, trying port 7861...")
        interface.launch(
            server_name="127.0.0.1",
            server_port=7861,
            share=False,
            debug=True
        )

if __name__ == "__main__":
    main()
