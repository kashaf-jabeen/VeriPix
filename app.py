import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from datetime import datetime
from gradcam_utils import generate_gradcam

# 1. Custom Theme & Page Config
st.set_page_config(page_title='VeriPix', page_icon='🔍', layout='wide')

st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #00ADB5; 
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Session State Initialization (History & Processed IDs Set)
if "history" not in st.session_state:
    st.session_state.history = []
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()

# 3. Sidebar
with st.sidebar:
    st.title("🔍 VeriPix")
    st.markdown("### How it works")
    st.info("Upload up to 5 images to detect if they are REAL or AI-GENERATED (FAKE). The system provides a Grad-CAM heatmap to explain its decision visually.")
    
    st.markdown("### Model Architecture")
    st.success("""
    - **Backbone:** ResNet-18
    - **Dataset:** CIFAKE Benchmark
    - **Accuracy:** Optimized for High Precision
    - **XAI:** Grad-CAM Layer-4
    """)
    
    st.write("---")
    if st.button("Clear Analysis History", use_container_width=True):
        st.session_state.history = []
        st.session_state.processed_ids = set()
        st.rerun()

# 4. Model Loader
@st.cache_resource
def load_model(model_path="veripix_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. UI Layout with Tabs
tab_analyze, tab_history = st.tabs(["🔍 Analyze Images", "📜 Analysis History"])

# ----------------- TAB 1: ANALYZE -----------------
with tab_analyze:
    st.header("Batch Image Analysis")
    st.write("Select multiple images for automated deepfake detection.")
    
    uploaded_files = st.file_uploader("Upload Image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("⚠️ Max 5 images allowed. Processing only the first 5 images.")
            uploaded_files = uploaded_files[:5]
            
        st.write("---")
        
        for idx, file in enumerate(uploaded_files):
            # Unique identifier for each uploaded image
            file_id = f"{file.name}_{file.size}"
            
            image = Image.open(file).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
            classes = ['FAKE', 'REAL']
            result = classes[pred.item()]
            score = conf.item() * 100
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate Heatmap
            heatmap = generate_gradcam(model, input_tensor, image)
            
            # Append to history ONLY if not already processed in this session
            if file_id not in st.session_state.processed_ids:
                entry = {
                    "Filename": file.name, 
                    "Verdict": result, 
                    "Confidence": f"{score:.2f}%", 
                    "Timestamp": timestamp
                }
                st.session_state.history.append(entry)
                st.session_state.processed_ids.add(file_id)
                
            # Render Cards
            with st.container(border=True):
                st.markdown(f"#### Target: `{file.name}`")
                col1, col2, col3 = st.columns([1, 1.5, 1])
                
                with col1:
                    st.image(image, caption="Original Input", use_container_width=True)
                    
                with col2:
                    st.write("")
                    st.metric(
                        label="Classification Verdict", 
                        value=result, 
                        delta=f"{score:.2f}% Confidence", 
                        delta_color="normal" if result == "REAL" else "inverse"
                    )
                    st.progress(int(score), text="Confidence Level")
                    
                    if result == "FAKE":
                        st.error("⚠️ AI-Generated features detected in image texture/pixels.")
                    else:
                        st.success("✅ Image structure appears authentic.")
                        
                with col3:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)

# ----------------- TAB 2: HISTORY -----------------
with tab_history:
    st.header("Session Analytics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Processed", len(df))
        m2.metric("Flagged Fake 🔴", len(df[df['Verdict'] == 'FAKE']))
        m3.metric("Verified Real 🟢", len(df[df['Verdict'] == 'REAL']))
        
        st.write("---")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No images analyzed yet. Upload files in the 'Analyze Images' tab to view history.")