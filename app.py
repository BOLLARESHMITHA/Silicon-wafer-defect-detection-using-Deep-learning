import os
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io

# ───────────────── CONFIG ─────────────────
st.set_page_config(
    page_title="Wafer Defect Inspector",
    page_icon="🔬",
    layout="wide"
)

IMG_SIZE = 64
DEVICE = torch.device("cpu")

# ───────────────── CLASS NAMES ─────────────────
@st.cache_resource
def load_class_names():
    path = "wafer_imgs/class_names.npy"
    if os.path.exists(path):
        return list(np.load(path, allow_pickle=True))
    return ["Center","Donut","Edge-Loc","Edge-Ring",
            "Loc","Near-full","Random","Scratch","none"]

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# ───────────────── MODEL ─────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,d_model,3,padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h = img_size//4
        seq_len = feat_h*feat_h

        self.pos = nn.Parameter(torch.zeros(1,seq_len,d_model))
        encoder = nn.TransformerEncoderLayer(d_model=d_model,nhead=4,batch_first=True)
        self.trans = nn.TransformerEncoder(encoder, num_layers=2)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model,256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.cnn(x)
        B,C,H,W = x.shape
        x = x.view(B,C,-1).permute(0,2,1)
        x = x + self.pos[:, :x.size(1), :]
        x = self.trans(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ───────────────── LOAD MODEL ─────────────────
@st.cache_resource
def load_model():
    path = "best_hybrid.pth"
    if not os.path.exists(path):
        return None, "❌ best_hybrid.pth not found"

    model = HybridCNNTransformer(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model, None

model, err = load_model()
if err:
    st.error(err)
    st.stop()

# ───────────────── PREPROCESS (FIXED) ─────────────────
def preprocess(img):
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.stack([img]*3, axis=-1)
    return img   # ✅ FIXED

# ───────────────── CHART ─────────────────
def plot_probs(probs):
    fig, ax = plt.subplots(figsize=(6,3))
    ax.barh(CLASS_NAMES, probs)
    ax.set_xlim(0,1)
    ax.set_title("Class Probabilities")
    return fig

# ───────────────── UI ─────────────────
st.title("🔬 Silicon Wafer Defect Inspector")

file = st.file_uploader("Upload wafer image", type=["png","jpg","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image")

    img = preprocess(image)
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred = CLASS_NAMES[pred_idx]
    conf = probs[pred_idx]

    st.success(f"Prediction: {pred}")
    st.info(f"Confidence: {conf:.2%}")

    st.pyplot(plot_probs(probs))
