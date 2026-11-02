from PIL import Image
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.title("Detektor Anime AI (Feature-Based OOD)")


# 1. Load Model & Modifikasi Ekstraksi Fitur
@st.cache_resource
def load_model_with_feature_extractor():
    # Load model EfficientNet-B0 utama
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(
        torch.load('best_model_efficientnet.pth', map_location='cpu')
    )
    model.eval()

    # Buat feature extractor (mengambil output sebelum classifier layer / 1280 dimensi)
    feature_extractor = torch.nn.Sequential(
        model.features, model.avgpool, torch.nn.Flatten()
    )

    return model, feature_extractor


model, feature_extractor = load_model_with_feature_extractor()
class_names = ['AI-generated', 'Human-made']

# 2. Vektor Acuan Anime (Centroid / Prototype Fitur Anime)
# Dapat dibuat dari rata-rata 10-20 sampel anime, atau dihitung secara dinamis.
# Vektor acuan ini merepresentasikan karakteristik domain gaya seni anime.
ANIME_CENTROID = None  # Akan diinisialisasi pada eksekusi pertama

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader(
    'Pilih gambar anime', type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # A. Ekstrak Vektor Fitur 1280-Dimensi dari Input
        features = feature_extractor(tensor)  # Shape: [1, 1280]

        # Norm vektor untuk Cosine Similarity
        norm_features = torch.nn.functional.normalize(features, p=2, dim=1)

        # B. Dapatkan Logits & Softmax untuk Prediksi Utama
        logits = model(tensor)
        probs = torch.softmax(logits[0], dim=0)
        idx = torch.argmax(probs).item()
        confidence = probs[idx].item() * 100

        # C. Evaluasi Deviasi Fitur (Norm-Based OOD Metric)
        # Gambar di luar domain (foto nyata/objek lain) memiliki magnitudo dan norma fitur yang menyimpang
        feature_norm = torch.norm(features, p=2).item()

    # THRESHOLD NORMA FITUR
    # Gambar anime biasanya memiliki norma fitur yang stabil pada rentang tertentu.
    # Jika feature_norm berada di luar rentang wajar (terlalu rendah/tinggi), tandai sebagai OOD.
    st.write(f'🔍 **Magnitudo Fitur (1280-D):** `{feature_norm:.2f}`')

    # Contoh Thresholding berdasarkan Norma Vektor Fitur
    if feature_norm < 12.0 or feature_norm > 28.0:
        st.warning(
            '⚠️ **Deteksi Out-of-Distribution:** Gambar terindikasi **Bukan Anime / Luar Domain**.'
        )
        if st.button('Tetap Paksa Prediksi'):
            st.info(f'**Prediksi:** {class_names[idx]} ({confidence:.2f}%)')
    else:
        st.success('✅ **Gambar Terverifikasi Domain Anime**')
        st.write(f'**Prediksi:** {class_names[idx]}')
        st.write(f'**Keyakinan:** {confidence:.2f}%')
