import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from .config import CONFIDENCE_THRESHOLD, UNCERTAINTY_THRESHOLD, MC_SAMPLES_DEFAULT

# OpenCV is optional; handle gracefully if not installed
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

# -----------------------------
# Configuration & Styling (must be first Streamlit calls)
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules (relative)
try:
    from .model import CNN_MLP_Model, CNN_GAP_Model
    from .dataset import data_transforms
    from .explain import GradCAM
except ImportError as e:
    st.error(f"Error importing modules: {e}. Ensure the 'src' package is installed.")
    st.stop()

# Custom CSS for "Advanced" look
st.markdown("""
<style>
    .main {
        background-color: #3542b8;
    }
    .stButton\
button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    h1 { color: #ffffff; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model Loading (Cached)
# -----------------------------
@st.cache_resource
def load_model(device):
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    # Prefer v2 checkpoint with improved architecture
    v2_candidates = ["best_model_v2.pth", "final_model_v2.pth"]
    v1_candidates = ["best_model.pth", "final_model.pth"]

    chosen_path = None
    model = None

    for fname in v2_candidates:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            chosen_path = path
            model = CNN_GAP_Model(num_classes=2)
            break

    if model is None:
        # fallback to v1
        for fname in v1_candidates:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                chosen_path = path
                model = CNN_MLP_Model(num_classes=2)
                break

    if model is None:
        # Nothing found; default to improved model with random weights
        st.warning("No model checkpoint found. Using CNN_GAP_Model with random weights.")
        model = CNN_GAP_Model(num_classes=2)
    else:
        try:
            state_dict = torch.load(chosen_path, map_location=device)
            model.load_state_dict(state_dict)
            st.success(f"Loaded model weights from {os.path.basename(chosen_path)}")
        except Exception as e:
            st.warning(f"Failed to load weights from {chosen_path}: {e}. Using random weights.")

    model.to(device)
    model.eval()
    return model

# -----------------------------
# Inference Logic
# -----------------------------
def _enable_dropout_only(module: nn.Module) -> None:
    """
    Put model in eval mode but enable only Dropout layers for MC Dropout.
    Keeps BatchNorm layers in eval to avoid batch-size=1 errors.
    """
    module.eval()
    for sub_module in module.modules():
        if isinstance(sub_module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            sub_module.train()
        # Explicitly keep BatchNorm in eval
        if isinstance(sub_module, nn.modules.batchnorm._BatchNorm):
            sub_module.eval()

def predict_mc(model, image, device, n_samples=30):
    """
    Perform Monte Carlo Dropout Inference.
    Returns: prediction class, mean confidence, uncertainty, and raw probabilities list
    """
    # Preprocess
    transform = data_transforms['test']
    img_tensor = transform(image).unsqueeze(0).to(device)

    # MC Inference
    _enable_dropout_only(model)  # Enable only dropout; keep BatchNorm in eval
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            preds.append(probs.unsqueeze(0))

    preds = torch.cat(preds, dim=0)  # Shape: [n_samples, 1, num_classes]

    mean_probs = preds.mean(dim=0)
    uncertainty = preds.var(dim=0)

    prob, pred_idx = torch.max(mean_probs, dim=1)
    class_name = ['NORMAL', 'PNEUMONIA'][pred_idx.item()]

    return class_name, prob.item(), uncertainty[0][pred_idx].item(), preds[:, 0, 1].cpu().numpy()  # Return PNEUMONIA probs

def apply_reject_option(pred_class: str, confidence: float, uncertainty_value: float, 
                        min_confidence: float, max_uncertainty: float):
    """
    Decide final label with a reject option (UNKNOWN) based on confidence/uncertainty thresholds.
    """
    is_unknown = (confidence < min_confidence) or (uncertainty_value > max_uncertainty)
    final_label = "UNKNOWN / Not a chest X-ray" if is_unknown else pred_class
    return final_label, is_unknown

# -----------------------------
# Visualization Logic
# -----------------------------
def plot_uncertainty_dist(probs):
    """Plot histogram of MC Dropout probabilities for Pneumonia class."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Uncertainty Distribution (MC Dropout)")
    ax.set_xlabel("Predicted Probability (Pneumonia)")
    ax.set_ylabel("Frequency")
    ax.axvline(np.mean(probs), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(probs):.2f}')
    ax.legend()
    return fig

def get_gradcam_heatmap(model, image, device):
    """Generate Grad-CAM heatmap."""
    # Re-preprocess for GradCAM which expects 4D input
    transform = data_transforms['test']
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Initialize GradCAM
    # Target the last conv layer. In CNN_MLP_Model it works best on conv2 or similar.
    grad_cam = GradCAM(model, model.conv2)

    heatmap = grad_cam(img_tensor)
    return heatmap

def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on original image."""
    if not _HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for heatmap overlay. Install 'opencv-python'.")
    img_np = np.array(image.convert("RGB"))
    img_cv = cv2.resize(img_np, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Swap BGR to RGB for matplotlib/streamlit
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = (heatmap_colored * alpha + img_cv * (1-alpha)).astype(np.uint8)
    return superimposed

# -----------------------------
# Main App Layout
# -----------------------------
def main():
    # Sidebar
    st.sidebar.title("🫁 Configuration")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.sidebar.markdown(f"**Device:** `{device}`")
    n_samples = st.sidebar.slider("MC Dropout Samples", min_value=10, max_value=100, value=MC_SAMPLES_DEFAULT, step=5)
    cam_alpha = st.sidebar.slider("Grad-CAM Opacity", 0.0, 1.0, 0.4)
    st.sidebar.markdown("### Reject Option")
    min_conf = st.sidebar.slider("Min confidence to accept", 0.5, 0.99, float(CONFIDENCE_THRESHOLD), 0.01)
    max_uncert = st.sidebar.slider("Max uncertainty to accept", 0.0, 0.2, float(UNCERTAINTY_THRESHOLD), 0.005)

    st.sidebar.markdown("---")
    st.sidebar.info("This app uses a CNN + MLP model with Monte Carlo Dropout to detect Pneumonia from Chest X-Rays and estimate uncertainty.")

    # Header
    st.title("Pneumonia Detection System")
    st.markdown("### Advanced Interface with Explainability & Uncertainty Estimation")

    # Layout
    col1, col2 = st.columns([1, 1.5])

    model = load_model(device)

    with col1:
        st.subheader("1. Upload X-Ray")
        uploaded_file = st.file_uploader("Choose a Chest X-Ray image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            analyze_btn = st.button("🔍 Analyze Image")
        else:
            # Placeholder to keep layout consistent
            st.info("Please upload an image to start.")
            analyze_btn = False

    if analyze_btn and uploaded_file:
        with st.spinner("Running Inference with Uncertainty Estimation..."):
            # Run Inference
            class_name, conf, uncert, probs_dist = predict_mc(model, image, device, n_samples)
            final_label, is_unknown = apply_reject_option(class_name, conf, uncert, min_conf, max_uncert)

            # Run Grad-CAM
            gradcam_error = None
            heatmap = None
            cam_image = None
            try:
                heatmap = get_gradcam_heatmap(model, image, device)
                cam_image = overlay_heatmap(image, heatmap, alpha=cam_alpha)
            except Exception as e:
                gradcam_error = str(e)

        # -----------------------------
        # Results Section
        # -----------------------------
        with col2:
            st.subheader("2. Analysis Results")

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class=\"metric-card\">
                    <h3>Prediction</h3>
                    <h2 style=\"color: {'#f39c12' if final_label.startswith('UNKNOWN') else ('#e74c3c' if final_label == 'PNEUMONIA' else '#27ae60')};\">{final_label}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class=\"metric-card\">
                    <h3>Confidence</h3>
                    <h2>{conf:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class=\"metric-card\">
                    <h3>Uncertainty</h3>
                    <h2>{uncert:.4f}</h2>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            if is_unknown:
                st.warning("The model is uncertain about this image (low confidence or high uncertainty). It may not be a valid chest X-ray or is out of distribution.")

            # Tabs for Visualizations
            tab1, tab2 = st.tabs(["🔥 Model Explanation (Grad-CAM)", "📊 Uncertainty Distribution"])

            with tab1:
                st.markdown("**Where is the model looking?** Red areas indicate high importance.")
                if cam_image is not None:
                    st.image(cam_image, caption=f"Grad-CAM Heatmap (Class: {class_name})", use_column_width=True)
                else:
                    st.warning(f"Grad-CAM unavailable: {gradcam_error}")

            with tab2:
                st.markdown("**How sure is the model?** Spread indicates uncertainty.")
                fig = plot_uncertainty_dist(probs_dist)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
