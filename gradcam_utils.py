import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, original_image_pil, target_category=None):
    # ResNet18 ke liye last convolutional layer target_layer hoti hai
    target_layers = [model.layer4[-1]]
    
    # GradCAM object initialize karein
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Target class specify karein (agar None hai toh top prediction auto-pick hogi)
    targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
    
    # Heatmap generate karein
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Image ko RGB 0-1 scale float numpy array mein convert karein
    img_resized = original_image_pil.resize((224, 224))
    rgb_img = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Original image ke upar heatmap overlay karein
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization