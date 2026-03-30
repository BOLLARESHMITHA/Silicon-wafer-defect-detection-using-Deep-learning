"""
Silicon Wafer Defect Detection — Streamlit App
CVR College of Engineering | CSE Data Science | Group 20
"""

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import os
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark semiconductor-industrial aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg-dark:    #0a0e1a;
    --bg-card:    #111827;
    --bg-card2:   #1a2235;
    --accent:     #00d4ff;
    --accent2:    #7c3aed;
    --accent3:    #10b981;
    --warn:       #f59e0b;
    --danger:     #ef4444;
    --text:       #e2e8f0;
    --text-muted: #64748b;
    --border:     #1e3a5f;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Main area */
.main .block-container { padding-top: 1.5rem; }
[data-testid="stAppViewContainer"] { background: var(--bg-dark) !important; }

/* Header banner */
.wafer-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 40%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.wafer-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,212,255,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(124,58,237,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.wafer-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
}
.wafer-header p {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin: 0;
    font-family: 'Space Mono', monospace;
}
.wafer-header .badge {
    display: inline-block;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.3);
    color: var(--accent);
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 0.6rem;
}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-card .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* Prediction result card */
.prediction-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 0.5rem;
}
.pred-class {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.02em;
}
.pred-conf {
    font-size: 1rem;
    color: var(--accent3);
    font-weight: 600;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding-left: 10px;
    margin: 1.5rem 0 1rem 0;
}

/* Info box */
.info-box {
    background: rgba(0,212,255,0.05);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* Class pill */
.class-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
    margin: 3px;
    font-weight: 600;
}

/* Defect color map */
.defect-Center    { background: rgba(0,212,255,0.15); color: #00d4ff; border: 1px solid rgba(0,212,255,0.4); }
.defect-Donut     { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.4); }
.defect-Edge-Loc  { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
.defect-Edge-Ring { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.defect-Loc       { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.defect-Random    { background: rgba(99,102,241,0.15); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.4); }
.defect-Scratch   { background: rgba(236,72,153,0.15); color: #f9a8d4; border: 1px solid rgba(236,72,153,0.4); }
.defect-Near-full { background: rgba(251,191,36,0.15); color: #fde68a; border: 1px solid rgba(251,191,36,0.4); }
.defect-none      { background: rgba(100,116,139,0.15);color: #94a3b8; border: 1px solid rgba(100,116,139,0.4);}

/* Progress bar override */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-muted) !important;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.15) !important;
    color: var(--accent) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15));
    border: 1px solid var(--border);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    border-radius: 8px;
}
.stButton > button:hover {
    border-color: var(--accent);
    background: rgba(0,212,255,0.2);
}

/* Matplotlib dark theme patch */
div[data-testid="stImage"] img { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 64
DEVICE     = torch.device("cpu")
SAVE_DIR   = "wafer_imgs"

# Defect class colors for charts
CLASS_COLORS = {
    "Center":    "#00d4ff",
    "Donut":     "#a78bfa",
    "Edge-Loc":  "#34d399",
    "Edge-Ring": "#fbbf24",
    "Loc":       "#f87171",
    "Random":    "#a5b4fc",
    "Scratch":   "#f9a8d4",
    "Near-full": "#fde68a",
    "none":      "#94a3b8",
}

DEFECT_INFO = {
    "Center":    "Cluster of defective dies at the wafer center. Often caused by polishing or CMP issues.",
    "Donut":     "Ring-shaped defect pattern excluding center. Associated with spin-coating non-uniformity.",
    "Edge-Loc":  "Defects localized at one edge zone. Caused by edge bead removal or edge exposure issues.",
    "Edge-Ring": "Full ring of defects along wafer edge. Related to mechanical handling or edge sealing.",
    "Loc":       "Localized cluster of defects in a specific region. Equipment contamination or particle deposit.",
    "Random":    "Randomly distributed defects across the wafer. Airborne particle contamination.",
    "Scratch":   "Linear or curved scratch pattern. Physical contact damage during handling or inspection.",
    "Near-full": "Defects covering most of the wafer surface. Severe process failure or contamination event.",
    "none":      "No defect detected. Wafer passes quality inspection.",
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION — must exactly match training
# ─────────────────────────────────────────────────────────────────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, img_size=64, d_model=128,
                 nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h   = img_size // 4
        seq_len  = feat_h * feat_h
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        feat = feat + self.pos_embed[:, :feat.size(1), :]
        feat = self.transformer(feat)
        return self.head(feat.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_class_names():
    cn_path = os.path.join(SAVE_DIR, "class_names.npy")
    if os.path.exists(cn_path):
        return list(np.load(cn_path, allow_pickle=True))
    return ["Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Random", "Scratch", "Near-full", "none"]

@st.cache_resource
def load_hybrid_model(num_classes):
    model = HybridCNNTransformer(num_classes=num_classes).to(DEVICE)
    if os.path.exists("best_hybrid.pth"):
        model.load_state_dict(torch.load("best_hybrid.pth", map_location=DEVICE))
        model.eval()
        return model, True
    return model, False

def preprocess(img_pil, size=IMG_SIZE):
    img_gray = np.array(img_pil.convert("L"), dtype=np.float32)
    img_gray = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_NEAREST)
    img_gray = img_gray / 255.0
    return np.stack([img_gray] * 3, axis=-1)

def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111827",
    "axes.facecolor":    "#111827",
    "axes.edgecolor":    "#1e3a5f",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e3a5f",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "legend.facecolor":  "#0d1321",
    "legend.edgecolor":  "#1e3a5f",
})


# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES      = load_class_names()
NUM_CLASSES      = len(CLASS_NAMES)
model, model_ok  = load_hybrid_model(NUM_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🔬</div>
        <div style='font-family: Space Mono, monospace; font-size:0.85rem;
                    color:#00d4ff; letter-spacing:0.1em;'>WAFER DEFECT</div>
        <div style='font-family: Space Mono, monospace; font-size:0.85rem;
                    color:#00d4ff; letter-spacing:0.1em;'>DETECTION</div>
    </div>
    <hr style='border-color:#1e3a5f;'>
    """, unsafe_allow_html=True)

    st.markdown("**📋 Project Info**")
    st.markdown(f"""
    <div class='info-box'>
    <b>College:</b> CVR College of Engineering<br>
    <b>Dept:</b> CSE — Data Science<br>
    <b>Group:</b> 20 (COEPB65)<br>
    <b>Members:</b> B. Reshmitha · K. Ushaswi<br>
    <b>Dataset:</b> WM811K (811,457 wafers)<br>
    <b>Classes:</b> {NUM_CLASSES} defect types
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🏗️ Architecture**")
    st.markdown("""
    <div class='info-box'>
    <b>Hybrid CNN-Transformer</b><br>
    CNN → local texture features<br>
    Transformer → global spatial context<br>
    <br>
    <b>Also trained:</b><br>
    • CNN Baseline<br>
    • ViT (from scratch)<br>
    • YOLOv11s (detection)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model status
    status_color = "#10b981" if model_ok else "#ef4444"
    status_text  = "Model loaded ✓" if model_ok else "Model not found"
    st.markdown(f"""
    <div style='background:rgba({"16,185,129" if model_ok else "239,68,68"},0.1);
                border:1px solid {status_color};
                border-radius:8px; padding:0.6rem 1rem;
                font-size:0.8rem; color:{status_color};
                font-family: Space Mono, monospace;'>
        ◉ {status_text}
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.warning("Place `best_hybrid.pth` and `wafer_imgs/class_names.npy` in the app directory.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🎨 Defect Classes**")
    for cls in CLASS_NAMES:
        css_cls = cls.replace(" ", "-")
        st.markdown(f"<span class='class-pill defect-{css_cls}'>{cls}</span>",
                    unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='wafer-header'>
    <h1>🔬 Silicon Wafer Defect Detection</h1>
    <p>Deep Learning-based semiconductor inspection using Hybrid CNN–Transformer architecture</p>
    <span class='badge'>WM811K Dataset</span>
    <span class='badge'>811K Wafer Maps</span>
    <span class='badge'>9 Defect Classes</span>
    <span class='badge'>Hybrid CNN-Transformer</span>
    <span class='badge'>YOLOv11</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Predict",
    "📊  Dataset & EDA",
    "📈  Training Results",
    "🎯  Model Comparison",
    "📖  About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Upload & Predict</div>", unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 1.4], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Drop a wafer map image here",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class='info-box' style='margin-top:0.8rem;'>
        <b>Supported formats:</b> PNG, JPG, JPEG<br>
        <b>Input:</b> 2D wafer map image (any size — auto-resized to 64×64)<br>
        <b>Model:</b> Hybrid CNN-Transformer<br>
        <b>Output:</b> Defect class + confidence + all probabilities
        </div>
        """, unsafe_allow_html=True)

        # Sample images hint
        sample_images = [f for f in os.listdir(".")
                         if f.endswith((".png", ".jpg")) and "wafer" in f.lower()]
        if sample_images:
            st.markdown("**Sample images found:**")
            sel = st.selectbox("Load a sample", ["— select —"] + sample_images,
                               label_visibility="collapsed")
            if sel != "— select —":
                uploaded = open(sel, "rb")

    with col_res:
        if uploaded is not None:
            img_pil = Image.open(uploaded)
            img_arr = preprocess(img_pil)

            # Run inference
            tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_idx   = int(np.argmax(probs))
            pred_cls   = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
            css_cls    = pred_cls.replace(" ", "-")

            # ── Prediction card ──────────────────────────────────────────────
            st.markdown(f"""
            <div class='prediction-card'>
                <div style='font-size:0.75rem; color:#64748b;
                            font-family:Space Mono,monospace;
                            text-transform:uppercase; letter-spacing:0.1em;
                            margin-bottom:0.4rem;'>DETECTED DEFECT</div>
                <div class='pred-class'>
                    <span class='class-pill defect-{css_cls}'>{pred_cls}</span>
                </div>
                <div class='pred-conf' style='margin-top:0.6rem;'>
                    Confidence: {confidence:.2%}
                </div>
                <div style='margin-top:0.8rem; font-size:0.83rem;
                            color:#94a3b8; line-height:1.5;'>
                    {DEFECT_INFO.get(pred_cls, "")}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='height:180px; border:2px dashed #1e3a5f;
                        border-radius:12px; display:flex;
                        align-items:center; justify-content:center;
                        color:#64748b; font-family:Space Mono,monospace;
                        font-size:0.85rem; text-align:center;'>
                Upload a wafer image<br>to see predictions
            </div>
            """, unsafe_allow_html=True)

    # ── Detailed results below ─────────────────────────────────────────────────
    if uploaded is not None:
        st.markdown("<hr>", unsafe_allow_html=True)
        det_col1, det_col2, det_col3 = st.columns([1, 1.2, 1.2], gap="large")

        with det_col1:
            st.markdown("<div class='section-header'>Input Image</div>",
                        unsafe_allow_html=True)
            # Show original + processed side by side
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(np.array(img_pil.convert("L")), cmap="plasma")
            axes[0].set_title("Uploaded", fontsize=9, color="#94a3b8")
            axes[0].axis("off")
            axes[1].imshow(img_arr[:, :, 0], cmap="plasma")
            axes[1].set_title("Preprocessed (64×64)", fontsize=9, color="#94a3b8")
            axes[1].axis("off")
            fig.patch.set_facecolor("#111827")
            plt.tight_layout(pad=0.5)
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)

        with det_col2:
            st.markdown("<div class='section-header'>Class Probabilities</div>",
                        unsafe_allow_html=True)
            # Horizontal bar chart
            sorted_pairs = sorted(zip(CLASS_NAMES, probs), key=lambda x: -x[1])
            names_sorted = [p[0] for p in sorted_pairs]
            probs_sorted = [p[1] for p in sorted_pairs]
            colors_sorted = [CLASS_COLORS.get(n, "#64748b") for n in names_sorted]

            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.barh(names_sorted[::-1], probs_sorted[::-1],
                           color=colors_sorted[::-1], alpha=0.85, height=0.6)
            for bar, val in zip(bars, probs_sorted[::-1]):
                ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va="center", fontsize=8, color="#e2e8f0")
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Probability", fontsize=8)
            ax.axvline(0.5, color="#1e3a5f", linewidth=1, linestyle="--")
            fig.patch.set_facecolor("#111827")
            plt.tight_layout()
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)

        with det_col3:
            st.markdown("<div class='section-header'>Confidence Gauge</div>",
                        unsafe_allow_html=True)
            # Donut gauge chart
            gauge_val   = confidence
            other_val   = 1 - gauge_val
            gauge_color = "#10b981" if gauge_val >= 0.85 else \
                          "#f59e0b" if gauge_val >= 0.60 else "#ef4444"

            fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(aspect="equal"))
            wedges, _ = ax.pie(
                [gauge_val, other_val],
                colors=[gauge_color, "#1e3a5f"],
                startangle=90,
                counterclock=False,
                wedgeprops=dict(width=0.35, edgecolor="#111827", linewidth=2)
            )
            ax.text(0, -0.05, f"{gauge_val:.1%}", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=gauge_color,
                    fontfamily="monospace")
            ax.text(0, -0.32, pred_cls, ha="center", va="center",
                    fontsize=9, color="#94a3b8", fontfamily="monospace")
            ax.text(0, 0.28, "CONFIDENCE", ha="center", va="center",
                    fontsize=7, color="#64748b", fontfamily="monospace",
                    fontweight="bold", letterspacing=2)
            fig.patch.set_facecolor("#111827")
            plt.tight_layout()
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET & EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Dataset Overview — WM811K</div>",
                unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("811,457", "Total Wafer Maps"),
        ("172,950", "Labeled Samples"),
        ("638,507", "Unlabeled Samples"),
        ("9", "Defect Classes"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    eda_col1, eda_col2 = st.columns([1.3, 1], gap="large")

    with eda_col1:
        st.markdown("<div class='section-header'>Class Distribution (Labeled Data)</div>",
                    unsafe_allow_html=True)
        # Show saved chart or generate
        if os.path.exists("class_distribution.png"):
            st.image("class_distribution.png", use_container_width=True)
        else:
            # Generate from known counts
            raw_counts = {
                "none": 147431, "Edge-Ring": 9680, "Loc": 6047,
                "Edge-Loc": 2735, "Center": 4294, "Scratch": 1193,
                "Random": 866, "Donut": 555, "Near-full": 149
            }
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            names = list(raw_counts.keys())
            vals  = list(raw_counts.values())
            colors = [CLASS_COLORS.get(n, "#64748b") for n in names]

            axes[0].bar(names, vals, color=colors, alpha=0.85)
            axes[0].set_title("Labeled Class Distribution", fontsize=11,
                              fontweight="bold", color="#e2e8f0")
            axes[0].set_ylabel("Count", fontsize=9)
            axes[0].tick_params(axis="x", rotation=45, labelsize=8)
            for i, v in enumerate(vals):
                axes[0].text(i, v + 500, f"{v:,}", ha="center",
                             fontsize=7, color="#94a3b8")

            axes[1].pie(
                [172950, 638507],
                labels=["Labeled\n~172K", "Unlabeled\n~638K"],
                colors=["#00d4ff", "#334155"],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 9, "color": "#e2e8f0"}
            )
            axes[1].set_title("Full 811K Dataset", fontsize=11,
                              fontweight="bold", color="#e2e8f0")
            fig.patch.set_facecolor("#111827")
            plt.tight_layout()
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)

    with eda_col2:
        st.markdown("<div class='section-header'>Class Imbalance Insight</div>",
                    unsafe_allow_html=True)
        # Imbalance table
        raw_counts_df = pd.DataFrame({
            "Class":    ["none", "Edge-Ring", "Loc", "Center", "Edge-Loc",
                         "Scratch", "Random", "Donut", "Near-full"],
            "Count":    [147431, 9680, 6047, 4294, 2735, 1193, 866, 555, 149],
            "Type":     ["Majority", "Normal", "Normal", "Normal", "Normal",
                         "Minority", "Minority", "Minority", "Minority 🔴"],
        })
        st.dataframe(
            raw_counts_df.style
            .apply(lambda row: [
                "background-color: rgba(239,68,68,0.1); color:#f87171"
                if row["Type"].startswith("Minority") else
                "background-color: rgba(0,212,255,0.05)"
                for _ in row
            ], axis=1),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div class='info-box' style='margin-top:0.8rem;'>
        <b>⚠️ Extreme Imbalance:</b><br>
        <code>none</code> class = 85% of data.<br>
        <code>Near-full</code> = only 149 samples.<br><br>
        <b>Solution applied:</b><br>
        • Heavy Albumentations augmentation for minority classes<br>
        • 3,000 samples per class after balancing<br>
        • Inverse-frequency weighted CrossEntropyLoss
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Sample Wafer Maps</div>",
                unsafe_allow_html=True)
    if os.path.exists("sample_wafers.png"):
        st.image("sample_wafers.png", use_container_width=True)
    else:
        st.info("Run the notebook to generate sample_wafers.png")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRAINING RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Training Results</div>",
                unsafe_allow_html=True)

    tr_col1, tr_col2 = st.columns([1, 1], gap="large")

    with tr_col1:
        st.markdown("**📉 Training & Validation Curves**")
        if os.path.exists("training_curves.png"):
            st.image("training_curves.png", use_container_width=True)
        else:
            st.info("Run the notebook to generate training_curves.png")

    with tr_col2:
        st.markdown("**📊 Augmentation Strategy**")
        aug_data = {
            "Class":         ["Center", "Edge-Ring", "Edge-Loc", "Loc",
                              "none", "Scratch", "Random", "Donut", "Near-full"],
            "Original":      [4294, 9680, 2735, 6047, 147431, 1193, 866, 555, 149],
            "After Balance": [3000] * 9,
            "Aug Type":      ["Standard", "Standard", "Standard", "Standard",
                              "Standard", "Heavy 🔴", "Heavy 🔴", "Heavy 🔴", "Heavy 🔴"],
        }
        aug_df = pd.DataFrame(aug_data)
        st.dataframe(aug_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class='info-box' style='margin-top:0.8rem;'>
        <b>Standard aug:</b> HorizontalFlip, VerticalFlip, Rotate(±30°),
        BrightnessContrast, GaussNoise, ShiftScaleRotate, ElasticTransform<br><br>
        <b>Heavy aug (minority classes):</b> All standard + wider Rotate(±45°),
        stronger ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout(6 holes)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Confusion matrices
    st.markdown("<div class='section-header'>Confusion Matrices</div>",
                unsafe_allow_html=True)
    cm_col1, cm_col2, cm_col3 = st.columns(3)

    for col, model_name, label in zip(
        [cm_col1, cm_col2, cm_col3],
        ["cnn", "hybrid", "vit"],
        ["CNN Baseline", "Hybrid CNN-Transformer", "ViT (Scratch)"]
    ):
        with col:
            st.markdown(f"**{label}**")
            cm_path = f"confusion_matrix_{model_name}.png"
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info(f"Run eval to generate\n{cm_path}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>All Models Comparison</div>",
                unsafe_allow_html=True)

    # Architecture cards
    arch_cols = st.columns(4, gap="small")
    archs = [
        ("🧠", "CNN Baseline",
         "3 Conv blocks + BatchNorm + AdaptiveAvgPool",
         "~174K params", "Classification", "#00d4ff"),
        ("🤖", "Vision Transformer",
         "ViT-Tiny: patch16, embed192, depth12, 3 heads",
         "~5.5M params", "Classification", "#a78bfa"),
        ("🔀", "Hybrid CNN-Transformer",
         "CNN local features → Transformer global context",
         "~558K params", "Classification", "#34d399"),
        ("🎯", "YOLOv11s",
         "Real-time detection with bounding boxes",
         "~9.4M params", "Detection + Location", "#fbbf24"),
    ]
    for col, (icon, name, desc, params, task, color) in zip(arch_cols, archs):
        with col:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left;'>
                <div style='font-size:1.5rem; margin-bottom:0.5rem;'>{icon}</div>
                <div style='font-family:Space Mono,monospace; font-size:0.82rem;
                            color:{color}; font-weight:700; margin-bottom:0.4rem;'>
                    {name}
                </div>
                <div style='font-size:0.78rem; color:#94a3b8; line-height:1.5;
                            margin-bottom:0.6rem;'>{desc}</div>
                <div style='font-size:0.72rem; color:#64748b;'>
                    ⚙️ {params}<br>📌 {task}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cmp_col1, cmp_col2 = st.columns([1.2, 1], gap="large")

    with cmp_col1:
        st.markdown("**📊 Accuracy & F1 Comparison Chart**")
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", use_container_width=True)
        else:
            # Generate placeholder chart
            models   = ["CNN", "ViT", "Hybrid"]
            accuracy = [0.901, 0.932, 0.965]
            f1       = [0.900, 0.930, 0.964]
            x = np.arange(len(models))
            w = 0.35
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - w/2, accuracy, w, label="Accuracy",   color="#00d4ff", alpha=0.85)
            ax.bar(x + w/2, f1,       w, label="F1 (macro)", color="#10b981", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(models)
            ax.set_ylim(0.8, 1.0); ax.set_ylabel("Score")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            ax.set_title("All Models — 9 Classes", fontsize=12,
                         fontweight="bold", color="#e2e8f0")
            for bar in ax.patches:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f"{bar.get_height():.3f}", ha="center", fontsize=9)
            fig.patch.set_facecolor("#111827")
            plt.tight_layout()
            st.image(fig_to_image(fig), use_container_width=True)
            plt.close(fig)

    with cmp_col2:
        st.markdown("**📋 Results Summary**")
        results_df = pd.DataFrame({
            "Model":       ["Hybrid CNN-Transformer ⭐", "ViT (scratch)", "CNN Baseline", "YOLOv11s"],
            "Task":        ["Classification", "Classification", "Classification", "Detection"],
            "Key Metric":  ["Accuracy ~96.5%", "Accuracy ~93.2%", "Accuracy ~90.1%", "mAP@0.5 ~96.1%"],
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class='info-box' style='margin-top:1rem;'>
        <b>Key Insight:</b><br>
        CNN/ViT/Hybrid → answer <b>WHAT</b> defect<br>
        YOLOv11 → answers <b>WHAT + WHERE</b> (bounding box)<br><br>
        Both are needed in a real fab inspection system.
        The classification model flags the type; YOLO
        localizes the exact defect region for repair.
        </div>
        """, unsafe_allow_html=True)

    # YOLO & Anomaly Detection
    st.markdown("<hr>", unsafe_allow_html=True)
    yolo_col, anom_col = st.columns([1, 1], gap="large")

    with yolo_col:
        st.markdown("<div class='section-header'>YOLOv11 Defect Detection</div>",
                    unsafe_allow_html=True)
        if os.path.exists("yolo_detections_grid.png"):
            st.image("yolo_detections_grid.png", use_container_width=True)
        else:
            st.info("Run YOLO inference to generate yolo_detections_grid.png")

        st.markdown("""
        <div class='info-box' style='margin-top:0.8rem;'>
        <b>YOLO Classes:</b> Center, Donut, Edge-Loc, Edge-Ring,
        Loc, Random, Scratch, Near-full (8 defect types)<br>
        <b>Model:</b> YOLOv11s (small) — 9.4M params<br>
        <b>Input:</b> 224×224 wafer images<br>
        <b>Output:</b> Bounding boxes + class + confidence
        </div>
        """, unsafe_allow_html=True)

    with anom_col:
        st.markdown("<div class='section-header'>EfficientAD — Anomaly Detection</div>",
                    unsafe_allow_html=True)
        if os.path.exists("anomaly_distribution.png"):
            st.image("anomaly_distribution.png", use_container_width=True)
        else:
            st.info("Run anomaly detection to generate anomaly_distribution.png")

        st.markdown("""
        <div class='info-box' style='margin-top:0.8rem;'>
        <b>Purpose:</b> Detect <i>unknown</i> defect types not in training set<br>
        <b>Method:</b> Autoencoder trained ONLY on normal wafers<br>
        <b>Detection rule:</b> High reconstruction error = anomaly<br>
        <b>Threshold:</b> mean + 2×std of normal reconstruction errors<br>
        <b>Use case:</b> New defect modes in production that weren't labeled
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    about_col1, about_col2 = st.columns([1.2, 1], gap="large")

    with about_col1:
        st.markdown("<div class='section-header'>About This Project</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Silicon wafers are the foundation of semiconductor manufacturing.
        Defects on wafer surfaces directly impact device yield and reliability.
        <br><br>
        This system automates defect detection using state-of-the-art deep learning —
        replacing slow manual inspection with a fast, accurate AI-powered pipeline.
        <br><br>
        <b>Dataset:</b> WM811K — 811,457 real wafer maps from semiconductor fabs,
        9 labeled defect categories, 638K unlabeled samples used for augmentation.
        <br><br>
        <b>Pipeline:</b> Raw pkl → label cleaning → Albumentations balancing
        (labeled + unlabeled) → DiskDataset (RAM-safe) → CNN / ViT / Hybrid training
        with weighted loss → YOLOv11 detection → EfficientAD anomaly detection
        → Streamlit deployment.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header' style='margin-top:1.5rem;'>Defect Class Guide</div>",
                    unsafe_allow_html=True)
        for cls, info in DEFECT_INFO.items():
            css_cls = cls.replace(" ", "-")
            st.markdown(
                f"<span class='class-pill defect-{css_cls}'>{cls}</span> "
                f"<span style='font-size:0.83rem; color:#94a3b8;'>{info}</span><br>",
                unsafe_allow_html=True
            )

    with about_col2:
        st.markdown("<div class='section-header'>Viva Q&A Quick Reference</div>",
                    unsafe_allow_html=True)
        qa_pairs = [
            ("Why use unlabeled data?",
             "638K unlabeled wafers used as augmentation pool — real device textures improve minority class diversity"),
            ("Why Albumentations?",
             "Realistic transforms (flip, rotate, elastic, dropout) prevent overfitting and improve generalization"),
            ("Why ViT from scratch?",
             "HuggingFace blocked in Colab; also proves architecture works without ImageNet pretraining"),
            ("CNN vs ViT vs Hybrid?",
             "CNN=local features, ViT=global attention, Hybrid=both combined via sequence modeling"),
            ("YOLO vs CNN/ViT?",
             "CNN/ViT → WHAT defect; YOLO → WHAT + WHERE (bounding box localization)"),
            ("What is EfficientAD?",
             "Trained on normal wafers only; high reconstruction error at test time = unknown defect"),
            ("Why DiskDataset?",
             "Prevents RAM crash — only one batch in memory vs 8GB X_all array"),
            ("Why weighted CrossEntropy?",
             "Near-full has 149 samples vs 147K none — inverse-frequency weights force model to learn rare classes"),
            ("Why AMP mixed precision?",
             "FP16 forward pass → 2× faster training, half VRAM usage on T4"),
            ("What are waferMap values?",
             "0 = no die, 1 = normal die, 2 = defective die"),
        ]
        for q, a in qa_pairs:
            with st.expander(f"❓ {q}"):
                st.markdown(f"""
                <div style='font-size:0.85rem; color:#94a3b8;
                            line-height:1.6; padding:0.3rem 0;'>
                    {a}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        <b>Team:</b><br>
        B. Reshmitha — 23B81A67G5<br>
        K. Ushaswi — 23B81A67J6<br><br>
        <b>Guide:</b> Dr. K. Sri Laxmi<br>
        <b>Institution:</b> CVR College of Engineering<br>
        <b>Dept:</b> CSE (Data Science)<br>
        <b>Semester:</b> B.Tech III Year II Sem<br>
        <b>Project:</b> Industry Oriented Mini Project
        </div>
        """, unsafe_allow_html=True)
