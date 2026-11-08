import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

# ==========================================
# 1. KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Sistem Klasifikasi Citra Anime 3-Kelas",
    page_icon="🎨",
    layout="wide"
)

# Pemetaan Indeks Sesuai Folder dataset_final
MODEL_PATH = "efficientnet_b0_final.pth"
CLASS_NAMES = [
    "AI-Anime (2025–2026)",       # Index 0
    "Human-Anime (Pre-2021)",     # Index 1
    "Real-Image (OOD Control)"    # Index 2
]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. LOAD MODEL (EFFICIENTNET-B0)
# ==========================================
@st.cache_resource
def load_trained_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        st.error(f"⚠️ File bobot model '{MODEL_PATH}' tidak ditemukan. Jalankan 'train_model.py' terlebih dahulu!")
        st.stop()
        
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# Transformasi Input (Standard ImageNet Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. FUNGSI INFERENSI PURE FORWARD PASS
# ==========================================
def predict_image(pil_img):
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(probs)
        
    return pred_idx, probs

# ==========================================
# 4. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.title("🎨 Sistem Klasifikasi Citra AI-Anime 3-Kelas")
st.markdown("""
Aplikasi ini melakukan uji inferensi klasifikasi citra menggunakan arsitektur **EfficientNet-B0** 
untuk membedakan **AI-Anime (2025–2026)**, **Human-Anime (Pre-2021)**, dan **Real-Image (OOD Control)**.
""")

st.divider()

# Sidebar Informasi Perangkat
with st.sidebar:
    st.header("⚙️ Status Perangkat")
    st.write(f"**Device:** `{DEVICE.type.upper()}`")
    if DEVICE.type == "cuda":
        st.write(f"**GPU:** `{torch.cuda.get_device_name(0)}`")
    st.write(f"**Model:** `EfficientNet-B0`")
    st.write(f"**Skema:** `3-Class Classification`")

# Area Unggah Gambar
uploaded_file = st.file_uploader("Unggah gambar uji (PNG, JPG, JPEG, WEBP):", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    with st.spinner("Memproses prediksi klasifikasi..."):
        pred_idx, probs = predict_image(img)

    st.subheader("📊 Hasil Prediksi Klasifikasi")
    
    col1, col2 = st.columns([1, 1.2])

    # Kolom 1: Display Gambar Input
    with col1:
        st.image(img, caption="Gambar Input Uji", use_container_width=True)

    # Kolom 2: Hasil & Distribusi Probabilitas
    with col2:
        st.markdown("### Kesimpulan Prediksi")
        
        hasil_prediksi = CLASS_NAMES[pred_idx]
        keyakinan = probs[pred_idx] * 100
        
        # Badge Status Klasifikasi
        if pred_idx == 0:
            st.error(f"**Hasil:** {hasil_prediksi}")
        elif pred_idx == 1:
            st.success(f"**Hasil:** {hasil_prediksi}")
        else:
            st.info(f"**Hasil:** {hasil_prediksi}")
            
        st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{keyakinan:.2f}%")
        st.write("---")
        
        # Progress Bar Distribusi 3 Kelas
        st.markdown("#### Distribusi Probabilitas Kelas:")
        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**")
            st.progress(float(probs[i]))
            st.caption(f"Nilai Kepastian: {probs[i]*100:.2f}%")
