# =============================================================================
# app.py — Silicon Wafer Defect Detection
# Deploy: streamlit run app.py
# Requires: best_hybrid.pth  +  wafer_imgs/class_names.npy  in same folder
# =============================================================================

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Inspector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark industrial theme ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0e1a;
    color: #c8d8e8;
}
.stApp { background-color: #0a0e1a; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}
section[data-testid="stSidebar"] * { color: #94b8d4 !important; }

/* ── Header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2f4a 50%, #0d1b2a 100%);
    border: 1px solid #1e4a7a;
    border-radius: 8px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;  left: -50%;
    width: 200%; height: 200%;
    background: repeating-linear-gradient(
        45deg,
        transparent,
        transparent 40px,
        rgba(30,90,150,0.04) 40px,
        rgba(30,90,150,0.04) 41px
    );
    pointer-events: none;
}
.hero-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.1rem;
    color: #48b0e8;
    letter-spacing: 0.06em;
    margin: 0 0 6px 0;
    text-shadow: 0 0 30px rgba(72,176,232,0.4);
}
.hero-sub {
    font-size: 0.9rem;
    color: #6a90b0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Cards ── */
.card {
    background: #0f1c2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.card-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #48b0e8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 14px;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 8px;
}

/* ── Prediction badge ── */
.pred-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0d3358, #1a5a8a);
    border: 1px solid #2a7ab8;
    border-radius: 6px;
    padding: 10px 22px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.6rem;
    color: #7dd3fc;
    letter-spacing: 0.08em;
    text-shadow: 0 0 20px rgba(125,211,252,0.5);
    margin: 8px 0;
}
.conf-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.4rem;
    color: #34d399;
    text-shadow: 0 0 20px rgba(52,211,153,0.4);
}
.conf-low  { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.4); }
.conf-mid  { color: #fbbf24; text-shadow: 0 0 20px rgba(251,191,36,0.4);  }

/* ── Progress bars ── */
.prob-row {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
    gap: 10px;
}
.prob-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #94b8d4;
    width: 100px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #1a2a3a;
    border-radius: 4px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #6a90b0;
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Status indicator ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399;
    margin-right: 8px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; box-shadow: 0 0 8px #34d399; }
    50%      { opacity:0.5; box-shadow: 0 0 16px #34d399; }
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1c2e !important;
    border: 2px dashed #1e4a7a !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #48b0e8 !important;
}

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── Metric ── */
[data-testid="metric-container"] {
    background: #0f1c2e;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 12px 16px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0e1a;
    border-bottom: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    color: #6a90b0 !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.1em;
}
.stTabs [aria-selected="true"] {
    color: #48b0e8 !important;
    border-bottom: 2px solid #48b0e8 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0f1c2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #94b8d4 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Info box ── */
.info-box {
    background: #0a1a2e;
    border-left: 3px solid #48b0e8;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #94b8d4;
    margin: 8px 0;
}
.warn-box {
    background: #1a150a;
    border-left: 3px solid #fbbf24;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #d4b894;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Constants & class metadata
# ═══════════════════════════════════════════════════════════════════════════
IMG_SIZE = 64
DEVICE   = torch.device("cpu")

# Defect descriptions shown in UI
DEFECT_INFO = {
    "Center":    ("⬤", "#f87171", "Defects concentrated at the wafer center. Usually caused by spin-coating or CMP issues."),
    "Donut":     ("◎", "#fb923c", "Ring-shaped pattern around center. Often linked to edge-exposure or etch uniformity."),
    "Edge-Loc":  ("◗", "#fbbf24", "Defects localised near the wafer edge. Linked to edge-bead removal or edge effects."),
    "Edge-Ring": ("○", "#a3e635", "Full or partial ring along the wafer perimeter. Common in deposition processes."),
    "Loc":       ("▪", "#34d399", "Localised cluster of defects anywhere on the wafer. Often a contamination event."),
    "Near-full": ("◼", "#22d3ee", "Almost the entire wafer surface affected. Indicates a systemic process failure."),
    "Random":    ("∷", "#818cf8", "Randomly distributed defects with no clear pattern. May indicate particle contamination."),
    "Scratch":   ("╱", "#c084fc", "Linear scratch pattern. Caused by mechanical contact during handling or CMP."),
    "none":      ("✓", "#94a3b8", "No defect detected. Wafer passes visual inspection."),
}


# ═══════════════════════════════════════════════════════════════════════════
# Load class names saved during training
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_class_names():
    path = "wafer_imgs/class_names.npy"
    if os.path.exists(path):
        return list(np.load(path, allow_pickle=True))
    # Fallback: alphabetical order matching LabelEncoder default
    return ["Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-full", "Random", "Scratch", "none"]

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# Model definition — must exactly match training
# ═══════════════════════════════════════════════════════════════════════════
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=9, img_size=64, d_model=128,
                 nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
        )
        feat_h   = img_size // 4
        seq_len  = feat_h * feat_h
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.cnn_backbone(x)
        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)
        feat = feat + self.pos_embed[:, :feat.size(1), :]
        feat = self.transformer(feat)
        return self.head(feat.mean(dim=1))


@st.cache_resource
def load_model():
    model = HybridCNNTransformer(num_classes=NUM_CLASSES).to(DEVICE)
    weights_path = "best_hybrid.pth"
    if not os.path.exists(weights_path):
        return None, f"❌ Weights file '{weights_path}' not found. Place it in the same folder as app.py."
    try:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval()
        return model, None
    except Exception as e:
        return None, f"❌ Failed to load weights: {e}"


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing — matches wafer_to_img() exactly
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(img_pil: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """colour / grayscale → gray → resize → normalise → 3-channel stack"""
    img_gray = np.array(img_pil.convert("L"), dtype=np.float32)
    img_gray = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_NEAREST)
    img_gray = img_gray / 255.0
    return np.stack([img_gray, img_gray, img_gray], axis=-1)   # (H, W, 3)


# ═══════════════════════════════════════════════════════════════════════════
# Chart helpers  (pure matplotlib → bytes, no plt.show())
# ═══════════════════════════════════════════════════════════════════════════
def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_probability_chart(probs: np.ndarray, class_names: list) -> bytes:
    colors = [DEFECT_INFO.get(c, ("", "#48b0e8", ""))[1] for c in class_names]
    sorted_pairs = sorted(zip(probs, class_names, colors), reverse=True)
    vals, names, cols = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0f1c2e")

    bars = ax.barh(range(len(names)), vals, color=cols, height=0.62,
                   edgecolor="none")
    # Glow effect: ghost bar behind
    ax.barh(range(len(names)), vals, color=cols, height=0.72,
            alpha=0.18, edgecolor="none")

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(min(v + 0.012, 0.97), i, f"{v:.1%}",
                va="center", ha="left", color="#c8d8e8",
                fontsize=8, fontfamily="monospace")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color="#94b8d4", fontfamily="monospace")
    ax.set_xlim(0, 1.13)
    ax.set_xlabel("Probability", color="#6a90b0", fontsize=8)
    ax.tick_params(colors="#6a90b0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.xaxis.label.set_color("#6a90b0")
    ax.tick_params(axis="x", colors="#6a90b0")
    ax.set_title("Class Probability Distribution", color="#48b0e8",
                 fontsize=10, pad=10, fontfamily="monospace")
    fig.tight_layout(pad=1.4)
    return _fig_to_bytes(fig)


def make_wafer_display(img_pil: Image.Image, pred_cls: str, confidence: float) -> bytes:
    """Show original upload + grayscale processed version side by side."""
    gray_arr = preprocess(img_pil, IMG_SIZE)[:, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor("#0a0e1a")

    for ax in axes:
        ax.set_facecolor("#0a0e1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e3a5f")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)

    # Original
    axes[0].imshow(np.array(img_pil.convert("RGB")))
    axes[0].set_title("Uploaded Image", color="#94b8d4",
                      fontsize=9, fontfamily="monospace")

    # Processed (what the model sees)
    color = DEFECT_INFO.get(pred_cls, ("", "#48b0e8", ""))[1]
    cmap  = matplotlib.colors.LinearSegmentedColormap.from_list(
        "wafer", ["#0a0e1a", color])
    axes[1].imshow(gray_arr, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title(f"Model Input  ({pred_cls})", color="#94b8d4",
                      fontsize=9, fontfamily="monospace")

    fig.suptitle(f"Confidence: {confidence:.1%}", color="#48b0e8",
                 fontsize=10, fontfamily="monospace", y=1.01)
    fig.tight_layout(pad=1.2)
    return _fig_to_bytes(fig)


def make_confidence_gauge(confidence: float) -> bytes:
    fig, ax = plt.subplots(figsize=(3.6, 1.8),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    theta_range = np.linspace(np.pi, 0, 200)
    # Background arc
    ax.plot(theta_range, [1] * 200, color="#1e3a5f", linewidth=12, solid_capstyle="round")

    # Value arc
    theta_val = np.linspace(np.pi, np.pi - confidence * np.pi, 100)
    if confidence >= 0.75:
        c = "#34d399"
    elif confidence >= 0.45:
        c = "#fbbf24"
    else:
        c = "#f87171"
    ax.plot(theta_val, [1] * 100, color=c, linewidth=12, solid_capstyle="round",
            alpha=0.9)

    ax.text(0, 0, f"{confidence:.1%}", ha="center", va="center",
            color=c, fontsize=18, fontweight="bold", fontfamily="monospace")
    ax.set_ylim(0, 1.4)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return _fig_to_bytes(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🔬 WAFER INSPECTOR")
    st.markdown("---")

    model, model_err = load_model()

    if model_err:
        st.error(model_err)
        st.stop()
    else:
        st.markdown('<span class="status-dot"></span>**Model loaded**', unsafe_allow_html=True)
        st.caption(f"{NUM_CLASSES} classes · HybridCNN-Transformer · CPU inference")

    st.markdown("---")
    st.markdown("**DEFECT CLASSES**")
    for cls in CLASS_NAMES:
        icon, color, _ = DEFECT_INFO.get(cls, ("·", "#94b8d4", ""))
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<span style="color:{color};font-size:1rem;">{icon}</span>'
            f'<span style="font-family:monospace;font-size:0.8rem;color:#94b8d4;">{cls}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**SETTINGS**")
    conf_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.05,
        help="Predictions below this are flagged as uncertain"
    )
    show_processed = st.checkbox("Show model input view", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#3a5a7a;font-family:monospace;">'
        'WM811K · Hybrid CNN-Transformer<br>'
        'Trained on 9 defect classes'
        '</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main layout
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🔬 SILICON WAFER DEFECT INSPECTOR</p>
  <p class="hero-sub">Hybrid CNN-Transformer · WM811K Dataset · 9 Defect Classes</p>
</div>
""", unsafe_allow_html=True)

# ── Upload + Results layout ───────────────────────────────────────────────────
left, right = st.columns([1, 1.4], gap="large")

with left:
    st.markdown('<div class="card-title">▸ UPLOAD WAFER IMAGE</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        img_pil = Image.open(uploaded)
        st.image(img_pil, use_container_width=True, caption="Uploaded wafer map")

        st.markdown(
            f'<div class="info-box">📐 Size: {img_pil.size[0]}×{img_pil.size[1]} px &nbsp;|&nbsp; '
            f'Mode: {img_pil.mode}</div>',
            unsafe_allow_html=True,
        )

with right:
    if not uploaded:
        st.markdown("""
        <div class="card" style="text-align:center;padding:60px 24px;">
          <div style="font-size:3rem;margin-bottom:16px;">🔬</div>
          <div style="font-family:'Share Tech Mono',monospace;color:#2a5a8a;font-size:1rem;letter-spacing:0.1em;">
            AWAITING INPUT
          </div>
          <div style="color:#2a4a6a;font-size:0.82rem;margin-top:10px;">
            Upload a wafer map image to begin analysis
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Run inference ──────────────────────────────────────────────────
        with st.spinner("Running inference..."):
            img_arr = preprocess(img_pil, IMG_SIZE)
            tensor  = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

        pred_idx   = int(np.argmax(probs))
        pred_cls   = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        icon, color, description = DEFECT_INFO.get(pred_cls, ("?", "#48b0e8", ""))

        # ── Primary result ─────────────────────────────────────────────────
        conf_class = "conf-value" if confidence >= 0.75 else ("conf-mid" if confidence >= 0.45 else "conf-low")
        low_conf_warning = confidence < conf_threshold

        st.markdown(f"""
        <div class="card">
          <div class="card-title">▸ ANALYSIS RESULT</div>
          <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
            <div>
              <div style="font-size:0.7rem;color:#3a6a8a;font-family:monospace;letter-spacing:0.15em;margin-bottom:6px;">PREDICTED CLASS</div>
              <div class="pred-badge">{icon}&nbsp; {pred_cls}</div>
            </div>
            <div>
              <div style="font-size:0.7rem;color:#3a6a8a;font-family:monospace;letter-spacing:0.15em;margin-bottom:4px;">CONFIDENCE</div>
              <div class="{conf_class}" style="font-family:'Share Tech Mono',monospace;font-size:2.2rem;">{confidence:.1%}</div>
            </div>
          </div>
          <div style="margin-top:14px;font-size:0.84rem;color:#6a90b0;border-top:1px solid #1e3a5f;padding-top:12px;">
            {description}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if low_conf_warning:
            st.markdown(
                f'<div class="warn-box">⚠️ Confidence ({confidence:.1%}) is below your threshold '
                f'({conf_threshold:.0%}). Consider re-examining this wafer manually.</div>',
                unsafe_allow_html=True,
            )

        # ── Tabs: Charts + Details ─────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["📊  PROBABILITIES", "🖼  IMAGE ANALYSIS", "📋  ALL SCORES"])

        with tab1:
            prob_chart = make_probability_chart(probs, CLASS_NAMES)
            st.image(prob_chart, use_container_width=True)

        with tab2:
            gauge_col, img_col = st.columns([1, 2])
            with gauge_col:
                gauge = make_confidence_gauge(confidence)
                st.image(gauge, use_container_width=True)
                st.caption("Confidence gauge")
            with img_col:
                if show_processed:
                    wafer_img = make_wafer_display(img_pil, pred_cls, confidence)
                    st.image(wafer_img, use_container_width=True)

        with tab3:
            # Sorted table of all class scores
            sorted_pairs = sorted(
                zip(CLASS_NAMES, probs.tolist()),
                key=lambda x: -x[1]
            )
            st.markdown('<div class="card-title">▸ ALL CLASS SCORES</div>', unsafe_allow_html=True)
            for rank, (cls, prob) in enumerate(sorted_pairs):
                icon_c, color_c, _ = DEFECT_INFO.get(cls, ("·", "#48b0e8", ""))
                bar_w  = int(prob * 260)
                is_top = rank == 0
                bg     = "#0d2a1e" if is_top else "transparent"
                border = f"border:1px solid {color_c}22;" if is_top else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:10px;padding:6px 10px;
                            background:{bg};border-radius:4px;{border}margin-bottom:4px;">
                  <span style="color:{color_c};width:16px;text-align:center;">{icon_c}</span>
                  <span style="font-family:monospace;font-size:0.8rem;color:#94b8d4;width:88px;">{cls}</span>
                  <div style="flex:1;height:6px;background:#1a2a3a;border-radius:3px;overflow:hidden;">
                    <div style="width:{bar_w}px;max-width:100%;height:100%;background:{color_c};
                                border-radius:3px;opacity:0.85;"></div>
                  </div>
                  <span style="font-family:monospace;font-size:0.78rem;color:#6a90b0;width:48px;text-align:right;">
                    {prob:.2%}
                  </span>
                  {'<span style="font-size:0.65rem;color:#34d399;font-family:monospace;margin-left:4px;">★ TOP</span>' if is_top else ''}
                </div>
                """, unsafe_allow_html=True)


# ── Bottom section: defect reference guide ────────────────────────────────────
st.markdown("---")
st.markdown('<div class="card-title" style="margin-top:8px;">▸ DEFECT CLASS REFERENCE GUIDE</div>',
            unsafe_allow_html=True)

cols = st.columns(3)
for i, cls in enumerate(CLASS_NAMES):
    icon, color, desc = DEFECT_INFO.get(cls, ("·", "#48b0e8", "No description."))
    with cols[i % 3]:
        st.markdown(f"""
        <div class="card" style="min-height:110px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
            <span style="font-size:1.4rem;color:{color};">{icon}</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.88rem;
                         color:{color};letter-spacing:0.06em;">{cls}</span>
          </div>
          <div style="font-size:0.78rem;color:#6a90b0;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
