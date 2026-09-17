import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks for gradients & activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cv2.resize(cam, (224, 224))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        return cam

def generate_gradcam_overlay(model, input_tensor, original_image_path, save_path):
    # Target ResNet layer (layer4)
    target_layer = model.layer4[-1]
    cam_generator = GradCAM(model, target_layer)
    cam = cam_generator.generate_cam(input_tensor)

    # Read original image using OpenCV
    img = cv2.imread(original_image_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_float = np.float32(img) / 255
    
    # Overlay heatmap on image
    cam_overlay = heatmap * 0.4 + img_float * 0.6
    cam_overlay = cam_overlay / np.max(cam_overlay)
    cv2.imwrite(save_path, np.uint8(255 * cam_overlay))