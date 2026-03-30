import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import os

# ───────────────── CONFIG ─────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="centered"
)

IMG_SIZE = 64
DEVICE = torch.device("cpu")

# ───────────────── LOAD CLASS NAMES ─────────────────
CLASS_PATH = "class_names.npy"

if not os.path.exists(CLASS_PATH):
    st.error("❌ class_names.npy not found")
    st.stop()

CLASS_NAMES = list(np.load(CLASS_PATH, allow_pickle=True))
NUM_CLASSES = len(CLASS_NAMES)

# ───────────────── MODEL (EXACT SAME AS TRAINING) ─────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=128,
                 nhead=4, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )

        feat_h = img_size // 4
        seq_len = feat_h * feat_h

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        feat = feat + self.pos_embed[:, :feat.size(1), :]
        feat = self.transformer(feat)
        feat = feat.mean(dim=1)
        return self.head(feat)

# ───────────────── LOAD MODEL ─────────────────
@st.cache_resource
def load_model():
    model_path = "best_hybrid.pth"

    if not os.path.exists(model_path):
        st.error("❌ best_hybrid.pth not found")
        st.stop()

    model = HybridCNNTransformer(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ───────────────── PREPROCESS (MATCH TRAINING) ─────────────────
def preprocess(img):
    img = img.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.stack([img]*3, axis=-1)
    return img

# ───────────────── UI ─────────────────
st.title("🔬 Silicon Wafer Defect Detection")
st.write(f"Model supports {NUM_CLASSES} defect classes")

uploaded_file = st.file_uploader("Upload Wafer Image", type=["png","jpg","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = preprocess(image)
    tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]

    st.success(f"Prediction: {pred_class}")
    st.info(f"Confidence: {confidence:.2%}")

    st.subheader("Class Probabilities")

    for cls, p in sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1]):
        st.progress(float(p), text=f"{cls}: {p:.2%}")
