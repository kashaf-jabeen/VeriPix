import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def load_trained_model(model_path="veripix_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ['FAKE', 'REAL']
    result = classes[predicted.item()]
    score = confidence.item() * 100

    return result, score

if __name__ == '__main__':
    model, device = load_trained_model("veripix_model.pth")
    
    # Check folder for any existing test image automatically
    test_dir = "dataset/test/REAL"
    if os.path.exists(test_dir) and os.listdir(test_dir):
        first_image = os.listdir(test_dir)[0]
        test_img = os.path.join(test_dir, first_image)
        
        result, score = predict_image(test_img, model, device)
        print(f"Testing Image: {test_img}")
        print(f"Prediction: {result} ({score:.2f}% confidence)")
    else:
        print("Please specify a valid image path in test_img variable.")