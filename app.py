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