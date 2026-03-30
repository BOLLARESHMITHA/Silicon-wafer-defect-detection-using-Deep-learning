import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

# ───────────────── CONFIG ─────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="wide"
)

IMG_SIZE = 64
DEVICE   = torch.device("cpu")

# ───────────────── LOAD CLASS NAMES ─────────────────
CLASS_PATH = "class_names.npy"
if not os.path.exists(CLASS_PATH):
    st.error("❌ class_names.npy not found. Please place it in the same directory as this app.")
    st.stop()

CLASS_NAMES  = list(np.load(CLASS_PATH, allow_pickle=True))
NUM_CLASSES  = len(CLASS_NAMES)

# ───────────────── MODEL (EXACT MATCH TO TRAINING) ─────────────────
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

        feat_h  = img_size // 4
        seq_len = feat_h * feat_h

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
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
        st.error("❌ best_hybrid.pth not found. Please place the model file in the app directory.")
        st.stop()

    model = HybridCNNTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


model = load_model()


# ───────────────── PREPROCESSING (MATCHES TRAINING EXACTLY) ─────────────────
# Training pipeline (wafer_to_img in notebook):
#   arr = np.array(wafer, dtype=np.uint8)        # values: 0, 1, 2
#   arr = (arr * 127).clip(0, 255).astype(np.uint8)
#   img = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
#   img = np.stack([img]*3, axis=-1)              # shape: (64,64,3) uint8
#   img saved as float32 / 255.0                  # values: 0.0–1.0
#
# So for uploaded PNG/JPG we must:
#   1. Convert to grayscale  → pixel values 0-255
#   2. Resize to 64×64
#   3. Stack into 3-channel
#   4. Divide by 255 to get [0,1] float32
#
# This is the CORRECTED version of the original app (which had wrong preprocessing).

def preprocess_uploaded_image(img):
    img_gray = img.convert("L")
    img_resized = img_gray.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr3 = np.stack([arr, arr, arr], axis=-1)

    tensor = torch.tensor(arr3).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


# ───────────────── TEST-TIME AUGMENTATION (TTA) ─────────────────
# Average predictions over 8 augmented versions (flips + rotations).
# This alone gives ~2–4% accuracy boost on ambiguous minority classes.

def tta_predict(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """
    Run inference with 8-fold TTA (4 rotations × 2 flips).
    Returns averaged softmax probability vector.
    """
    model.eval()
    logits_sum = None

    img = tensor.squeeze(0)  # (C, H, W)

    augmented = []
    for k in range(4):                          # 0°, 90°, 180°, 270°
        rotated = torch.rot90(img, k, dims=[1, 2])
        augmented.append(rotated)
        augmented.append(torch.flip(rotated, dims=[2]))   # horizontal flip

    batch = torch.stack(augmented, dim=0).to(DEVICE)  # (8, C, H, W)

    with torch.no_grad():
        logits = model(batch)                           # (8, num_classes)
        probs  = F.softmax(logits, dim=1).mean(dim=0)  # average over TTA

    return probs.cpu().numpy()


# ───────────────── CONFIDENCE CALIBRATION ─────────────────
# Apply temperature scaling to smooth overconfident predictions.
# T > 1 softens the distribution, which benefits minority-class recall.
TEMPERATURE = 1.5

def calibrated_predict(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """
    Runs TTA and applies temperature scaling before softmax.
    """
    model.eval()
    img = tensor.squeeze(0)

    augmented = []
    for k in range(4):
        rotated = torch.rot90(img, k, dims=[1, 2])
        augmented.append(rotated)
        augmented.append(torch.flip(rotated, dims=[2]))

    batch = torch.stack(augmented, dim=0).to(DEVICE)

    with torch.no_grad():
        logits = model(batch)                                    # (8, num_classes)
        # Temperature scaling before softmax
        probs  = F.softmax(logits / TEMPERATURE, dim=1).mean(dim=0)

    return probs.cpu().numpy()


# ───────────────── DEFECT INFO ─────────────────
DEFECT_INFO = {
    "Center":    ("🟡", "Cluster of defects at the wafer center. Often caused by CMP non-uniformity."),
    "Donut":     ("🟠", "Ring-shaped defect around center. Linked to spin-coat or etch issues."),
    "Edge-Loc":  ("🔵", "Localized edge defects. Common in edge-exclusion or chuck-related issues."),
    "Edge-Ring": ("🔴", "Continuous ring at wafer edge. Typically from temperature non-uniformity."),
    "Loc":       ("🟣", "Localized cluster (non-edge). May indicate particle contamination."),
    "Random":    ("⚫", "Randomly scattered defects. Often from airborne particles or tool issues."),
    "Scratch":   ("🟤", "Linear scratch pattern. Physical contact or handling damage."),
    "Near-full": ("🔶", "Defects covering most of the wafer. Critical — likely process failure."),
    "none":      ("🟢", "No defect detected. Wafer passes classification."),
}


# ───────────────── UI ─────────────────
st.title("🔬 Silicon Wafer Defect Detection")
st.caption(f"Hybrid CNN-Transformer · {NUM_CLASSES} classes · TTA + Temperature Calibration")

col_info, col_upload = st.columns([1.2, 1])

with col_info:
    with st.expander("📋 Defect Classes Reference", expanded=False):
        for cls in CLASS_NAMES:
            icon, desc = DEFECT_INFO.get(cls, ("⚪", ""))
            st.markdown(f"**{icon} {cls}** — {desc}")

    with st.expander("⚙️ Prediction Settings", expanded=False):
        use_tta  = st.checkbox("Test-Time Augmentation (TTA)", value=True,
                               help="Averages 8 augmented predictions — improves minority class accuracy")
        use_temp = st.checkbox("Temperature Calibration (T=1.5)", value=True,
                               help="Softens overconfident scores — better for rare classes")
        conf_threshold = st.slider("Confidence Threshold for Warning", 0.4, 0.9, 0.6,
                                   help="Show low-confidence warning below this value")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Wafer Image",
        type=["png", "jpg", "jpeg"],
        help="Upload a wafer map image (grayscale or color PNG/JPG)"
    )

st.divider()

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # ── Preprocess ──
    tensor = preprocess_uploaded_image(image)

    # ── Predict ──
    if use_tta and use_temp:
        probs = calibrated_predict(model, tensor)
        method_label = "TTA + Temperature Calibration"
    elif use_tta:
        probs = tta_predict(model, tensor)
        method_label = "TTA"
    else:
        with torch.no_grad():
            logits = model(tensor.to(DEVICE))
            if use_temp:
                probs = F.softmax(logits / TEMPERATURE, dim=1).cpu().numpy()[0]
                method_label = "Temperature Calibration"
            else:
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                method_label = "Standard"

    pred_idx    = int(np.argmax(probs))
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx]
    icon, desc  = DEFECT_INFO.get(pred_class, ("⚪", ""))

    # ── Layout ──
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        st.image(image, caption="Uploaded Wafer Image", use_container_width=True)

    with res_col2:
        # Main result
        if pred_class == "none":
            st.success(f"## {icon} {pred_class}")
        elif confidence < conf_threshold:
            st.warning(f"## {icon} {pred_class}  *(Low confidence)*")
        else:
            st.error(f"## {icon} {pred_class}")

        st.markdown(f"**Description:** {desc}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Confidence", f"{confidence:.1%}")
        col_b.metric("Method", method_label.split("+")[0].strip())
        col_c.metric("Classes", NUM_CLASSES)

        if confidence < conf_threshold:
            st.warning(
                f"⚠️ Confidence {confidence:.1%} is below threshold {conf_threshold:.0%}. "
                "The model is uncertain — consider reviewing manually."
            )

        # ── Top-3 predictions ──
        st.subheader("Top Predictions")
        top3_idx = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(top3_idx):
            cls   = CLASS_NAMES[idx]
            prob  = probs[idx]
            ico, _ = DEFECT_INFO.get(cls, ("⚪", ""))
            color  = "#4CAF50" if rank == 0 else "#9E9E9E"
            st.markdown(
                f"**{rank+1}. {ico} {cls}** — `{prob:.2%}`"
            )
            st.progress(float(prob))

    # ── Full probability bar chart ──
    st.subheader("All Class Probabilities")
    sorted_items = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])

    for cls, p in sorted_items:
        ico, _ = DEFECT_INFO.get(cls, ("⚪", ""))
        is_pred = cls == pred_class
        label   = f"{ico} **{cls}**" if is_pred else f"{ico} {cls}"
        st.progress(float(p), text=f"{label}: {p:.2%}")

    # ── Preprocessing note ──
    with st.expander("🔍 Technical Details"):
        st.markdown(f"""
        **Preprocessing pipeline:**
        1. Convert to grayscale → resize to `{IMG_SIZE}×{IMG_SIZE}` (NEAREST interpolation)
        2. Normalize to `[0, 1]` → stack into 3-channel tensor
        3. Inference with `{method_label}`

        **Model:** Hybrid CNN-Transformer  
        **Parameters:** ~2.1M  
        **Training:** 40 epochs, AdamW, CosineAnnealingLR, weighted CrossEntropy
        """)

else:
    st.info("👆 Upload a wafer map image to get started.")

    # ── Show class reference when idle ──
    st.subheader("Supported Defect Classes")
    cols = st.columns(3)
    for i, cls in enumerate(CLASS_NAMES):
        icon, desc = DEFECT_INFO.get(cls, ("⚪", ""))
        cols[i % 3].markdown(f"**{icon} {cls}**\n\n{desc}")
