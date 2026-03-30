import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os

# ───────────────── CONFIG ─────────────────
st.set_page_config(
    page_title="Wafer Defect Inspector",
    page_icon="🔬",
    layout="centered"
)

IMG_SIZE = 64
DEVICE = torch.device("cpu")

# ───────────────── STYLE ─────────────────
st.markdown("""
<style>
body {background-color: #0e1117; color: #ffffff;}
.big-title {font-size:28px; font-weight:700; color:#4FC3F7;}
.card {
    background:#1c1f26;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# ───────────────── CLASS NAMES ─────────────────
@st.cache_resource
def load_classes():
    path = "wafer_imgs/class_names.npy"
    if os.path.exists(path):
        return list(np.load(path, allow_pickle=True))
    return ["Center","Donut","Edge-Loc","Edge-Ring",
            "Loc","Near-full","Random","Scratch","none"]

CLASS_NAMES = load_classes()
NUM_CLASSES = len(CLASS_NAMES)

# ───────────────── MODEL ─────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.cnn(x)
        x = x.mean([2,3])
        return self.fc(x)

# ───────────────── LOAD MODEL ─────────────────
@st.cache_resource
def load_model():
    path = "best_hybrid.pth"
    if not os.path.exists(path):
        return None

    model = HybridCNNTransformer(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("❌ Model file not found (best_hybrid.pth)")
    st.stop()

# ───────────────── PREPROCESS ─────────────────
def preprocess(img):
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.stack([img]*3, axis=-1)
    return img

# ───────────────── UI ─────────────────
st.markdown('<div class="big-title">🔬 Wafer Defect Inspector</div>', unsafe_allow_html=True)
st.write("Upload a wafer image to detect defect type")

file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if file:
    image = Image.open(file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess(image)
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx]

    # ── RESULT CARD ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf:.2%}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PROBABILITY BARS ──
    st.subheader("Class Probabilities")

    for cls, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.progress(float(p), text=f"{cls}: {p:.2%}")
