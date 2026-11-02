from PIL import Image
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.title("Uji Coba Detektor Anime AI")


# 1. Load Model EfficientNet-B0
@st.cache_resource
def load_model():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(
        torch.load('best_model_efficientnet.pth', map_location='cpu')
    )
    model.eval()
    return model


model = load_model()
class_names = ['AI-generated', 'Human-made']

# 2. Preprocessing Gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# 3. Fungsi Energy-Based OOD Score
def compute_energy_score(logits, temperature=1.0):
    """Menghitung Energy Score dari logits PyTorch.

    E(x) = -T * log(sum(exp(logits / T)))
    - Semakin negatif nilainya (misal -6.0) -> In-Distribution (Anime)
    - Semakin mendekati nol/positif (misal -1.0 atau 0.5) -> Out-of-Distribution (Non-Anime)
    """
    logits_scaled = logits / temperature
    energy = -temperature * torch.logsumexp(logits_scaled, dim=1).item()
    return energy


# --- AMBANG BATAS (THRESHOLD) OOD ---
# Sesuaikan angka ini sesuai hasil kalibrasi sampel anime kamu (acuan awal: -3.0)
ENERGY_THRESHOLD = -0.82

# 4. Upload & Prediksi
uploaded_file = st.file_uploader(
    'Pilih gambar anime', type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)

    # Preprocessing Tensor
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Dapatkan keluaran mentah (logits) dari EfficientNet
        logits = model(tensor)

        # Hitung Energy Score untuk OOD Detection
        energy_score = compute_energy_score(logits)

        # Hitung Probabilitas Softmax untuk Klasifikasi Biner
        probs = torch.softmax(logits[0], dim=0)
        idx = torch.argmax(probs).item()
        confidence = probs[idx].item() * 100

    # Pengecekan Kondisi OOD
    is_ood = energy_score > ENERGY_THRESHOLD

    if is_ood:
        st.warning(
            f"⚠️ **Peringatan Out-of-Distribution (OOD):** Gambar terindikasi **Non-Anime / Luar Domain**.\n\n"
            f"*(Energy Score: `{energy_score:.2f}` | Threshold: `{ENERGY_THRESHOLD}`)*"
        )
        # Tombol Override: Pengguna tetap bisa memaksa prediksi jika yakin gambar tersebut anime
        if st.button("Tetap Lanjutkan Prediksi Model"):
            st.success("✅ **Hasil Prediksi Forced Override:**")
            st.write(f"**Prediksi:** {class_names[idx]}")
            st.write(f"**Keyakinan:** {confidence:.2f}%")
    else:
        st.success("✅ **Validasi Sukses:** Gambar terverifikasi domain Anime!")
        st.write(f"**Prediksi:** {class_names[idx]}")
        st.write(f"**Keyakinan:** {confidence:.2f}%")
