import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import build_model
from src.config import IMG_SIZE

# --- SETTINGS ---
MODEL_PATH = "models/best_model.pth"
LABELS_PATH = "models/labels.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD LABELS ---
with open(LABELS_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- LOAD MODEL ---
@st.cache_resource 
def load_trained_model():
    model = build_model(num_classes=len(class_names), device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_trained_model()

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- UI DESIGN ---
st.set_page_config(page_title="Plant Doctor", layout="centered")
st.title("🌱 AI Plant Disease Doctor")

# Sidebar for options
st.sidebar.header("Options")
input_mode = st.sidebar.radio("Choose Input Mode:", ["Upload Image", "Take a Photo"])

source = None
if input_mode == "Upload Image":
    source = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    source = st.camera_input("Scan a leaf")

if source is not None:
    # Display and Process
    image = Image.open(source).convert('RGB')
    if input_mode == "Upload Image":
        st.image(image, caption='Target Leaf', use_container_width=True)
    
    if st.button("Analyze Leaf"):
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, index = torch.max(probabilities, 0)
            
        result = class_names[index]
        
        # Display Results with UI feedback
        st.divider()
        st.subheader(f"Diagnosis: {result}")
        st.progress(float(confidence.item()))
        st.write(f"**Confidence Score:** {confidence.item():.2%}")
        
        if "healthy" in result.lower():
            st.success("✅ This plant looks healthy! Keep up the good work.")
        else:
            st.error("⚠️ Disease Detected. You may need to treat this plant.")