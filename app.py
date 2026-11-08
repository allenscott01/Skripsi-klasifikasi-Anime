import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Detektor AI-Anime | EfficientNet-B0",
    page_icon="🎨",
    layout="wide"
)

MODEL_PATH = "efficientnet_b0_final.pth"
CLASS_NAMES = ["AI-Anime (2025-2026)", "Human-Anime (Pre-2021)", "Real-Image"]
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD MODEL METODE CACHING
# ==========================================
@st.cache_resource
def load_trained_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        st.error(f"File model '{MODEL_PATH}' tidak ditemukan!")
        st.stop()
        
    model = model.to(DEVICE)
    model.eval()
    return model

# Load model ke memori
model = load_trained_model()

# ==========================================
# 3. FUNGSI PREPROCESSING & GRAD-CAM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_and_predict(pil_img):
    resized_orig = pil_img.resize((224, 224))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Hooks untuk ekstraksi fitur Grad-CAM
    activations = []
    gradients = []

    def hook_act(module, input, output):
        activations.append(output)

    def hook_grad(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer = model.features[-1]
    h1 = target_layer.register_forward_hook(hook_act)
    h2 = target_layer.register_full_backward_hook(hook_grad)

    # Forward pass
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
    pred_idx = np.argmax(probs)

    # Backward pass untuk Grad-CAM
    model.zero_grad()
    score = output[0, pred_idx]
    score.backward()

    # Lepas hook
    h1.remove()
    h2.remove()

    # Perhitungan Heatmap
    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(resized_orig), 0.6, heatmap, 0.4, 0)

    return resized_orig, overlay, pred_idx, probs

# ==========================================
# 4. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.title("🛡️ Sistem Deteksi Citra AI-Anime 3-Kelas")
st.markdown("""
Aplikasi ini memanfaatkan arsitektur **EfficientNet-B0** dan teknik **Explainable AI (Grad-CAM)** 
untuk mengidentifikasi artefak mikro/sub-piksel pada gambar buatan AI modern (2025–2026), lukisan manusia (Pre-2021), maupun gambar real sebagai kontrol Out-of-Distribution.
""")

st.divider()

# Sidebar Informasi & Pengaturan
with st.sidebar:
    st.header("⚙️ Informasi Sistem")
    st.write(f"**Device Komputasi:** `{DEVICE.type.upper()}`")
    st.write(f"**Model:** `EfficientNet-B0`")
    st.write(f"**Skema Klasifikasi:** `3 Kelas`")
    st.divider()
    st.info("Unggah file gambar berformat PNG, JPG, JPEG, atau WEBP untuk memulai analisis.")

# Area Upload File
uploaded_file = st.file_uploader("Pilih atau tarik gambar ke area ini:", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner("Menganalisis fitur sub-piksel dan mengenerate Grad-CAM..."):
        resized_orig, overlay, pred_idx, probs = process_and_predict(img)

    st.subheader("📊 Hasil Analisis Model")
    
    # Menampilkan Kolom
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        st.image(resized_orig, caption="Gambar Input (224x224)", use_container_width=True)

    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap (Fokus Atensi)", use_container_width=True)

    with col3:
        st.markdown("### Probabilitas Prediksi")
        
        # Tampilkan Metric Kesimpulan Utama
        hasil_prediksi = CLASS_NAMES[pred_idx]
        keyakinan = probs[pred_idx] * 100
        
        if pred_idx == 0:
            st.error(f"**Hasil:** {hasil_prediksi}")
        elif pred_idx == 1:
            st.success(f"**Hasil:** {hasil_prediksi}")
        else:
            st.info(f"**Hasil:** {hasil_prediksi}")
            
        st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{keyakinan:.2f}%")
        st.write("---")
        
        # Progress Bar Distribusi Probabilitas
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**")
            st.progress(float(probs[i]))
            st.caption(f"Kepastian: {probs[i]*100:.2f}%")
