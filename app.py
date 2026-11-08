import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
from torchvision import models, transforms

# ==========================================
# 1. KONFIGURASI & LOAD MODEL
# ==========================================
st.set_page_config(page_title="Detektor Forensik AI-Anime", page_icon="🛡️", layout="wide")

MODEL_PATH = "efficientnet_b0_final.pth"
CLASS_NAMES = ["AI-Anime", "Human-Anime", "Real-Image (OOD)"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"File bobot '{MODEL_PATH}' tidak ditemukan. Silakan jalankan pelatihan terlebih dahulu.")
        st.stop()
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 2. INFERENSI & GRAD-CAM
# ==========================================
def predict_and_cam(pil_img):
    img_resized = pil_img.resize((224, 224))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    activations, gradients = [], []
    def hook_act(module, input, output): activations.append(output)
    def hook_grad(module, grad_in, grad_out): gradients.append(grad_out[0])

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(hook_act)
    h2 = target_layer.register_full_backward_hook(hook_grad)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))

    model.zero_grad()
    output[0, pred_idx].backward()

    h1.remove()
    h2.remove()

    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)

    return img_resized, overlay, pred_idx, probs

# ==========================================
# 3. ANTARMUKA DASHBOARD
# ==========================================
st.title("🛡️ Detektor Forensik Citra AI-Anime")
st.caption(f"Perangkat Komputasi Aktif: {DEVICE.type.upper()}")

uploaded_file = st.file_uploader("Unggah gambar uji (PNG, JPG, JPEG, WEBP):", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    orig_img, cam_overlay, pred_idx, probs = predict_and_cam(img)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(orig_img, caption="Gambar Input", use_container_width=True)

    with col2:
        st.image(cam_overlay, caption="Heatmap Grad-CAM", use_container_width=True)

    with col3:
        st.subheader("Hasil Analisis")
        st.metric("Kategori Terdeteksi", CLASS_NAMES[pred_idx], f"{probs[pred_idx]*100:.2f}% Confidence")
        st.divider()
        
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**: {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))
