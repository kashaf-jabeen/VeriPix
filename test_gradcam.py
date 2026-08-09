import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from gradcam_utils import generate_gradcam

def load_trained_model(model_path="veripix_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

if __name__ == '__main__':
    model, device = load_trained_model("veripix_model.pth")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Sample images test karne ke liye directories
    test_folders = ["dataset/test/REAL", "dataset/test/FAKE"]

    for folder in test_folders:
        if os.path.exists(folder) and os.listdir(folder):
            img_name = os.listdir(folder)[0]
            img_path = os.path.join(folder, img_name)

            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            # Grad-CAM Heatmap Generate karein
            cam_image = generate_gradcam(model, input_tensor, pil_img)

            # Result Save karein
            save_name = f"gradcam_{os.path.basename(folder)}.png"
            plt.imsave(save_name, cam_image)
            print(f"Heatmap generated successfully for {folder}: Saved as '{save_name}'")