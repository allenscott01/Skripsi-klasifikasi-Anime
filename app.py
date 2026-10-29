import cv2
from PIL import Image
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.title("Uji Coba Detektor Anime AI")


# --- VALIDASI INPUT NON-ANIME (OPENCV) ---
def is_anime_image(pil_image):
    # Konversi gambar PIL ke NumPy array
    img_np = np.array(pil_image)

    # Ubah warna dari RGB ke Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Deteksi Garis Tepi (Canny Edge Detection)
    edges = cv2.Canny(gray, 100, 200)

    # Hitung kerapatan piksel garis (Edge Density)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = np.sum(edges > 0) / total_pixels

    # Batas toleransi kerapatan garis anime (dapat disesuaikan)
    if edge_density < 0.025 or edge_density > 0.22:
        return False
    return True


# 1. Load Model ringkas
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
# Urutan label sesuai abjad nama folder (AI = 0, Human = 1)
class_names = ['AI-generated', 'Human-made']

# 2. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. Upload & Prediksi
uploaded_file = st.file_uploader(
    'Pilih gambar anime', type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)

    # Saringan Awal: Cek apakah gambar bergaya anime
    if not is_anime_image(image):
        st.error(
            '**Gambar Terdeteksi Bukan Anime :'
        )
    else:
        # Proses prediksi PyTorch jika gambar lolos saringan
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs[0], dim=0)
            idx = torch.argmax(probs).item()

        # Tampilkan hasil
        st.success(' Gambar terverifikasi bergaya anime!')
        st.write(f'**Prediksi:** {class_names[idx]}')
        st.write(f'**Keyakinan:** {probs[idx].item() * 100:.2f}%')
