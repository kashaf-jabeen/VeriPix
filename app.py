import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from gradcam_utils import generate_gradcam

# Page Setup
st.set_page_config(
    page_title="VeriPix - AI Image Detector",
    page_icon="🔍",
    layout="centered"
)

# Title and Description
st.title("🔍 VeriPix: AI Image Detection & Explainability")
st.write(
    "Upload an image to check whether it is **REAL** or **AI-GENERATED (FAKE)**. "
    "This tool also provides a Grad-CAM heatmap to highlight decision regions."
)

# Load Trained Model (Cached so it loads only once)
@st.cache_resource
def load_model(model_path="veripix_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Architecture
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

# Call model loader
try:
    model, device = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")