import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Klasifikasi Citra Anime", page_icon="🖼️", layout="centered")
st.title("🖼️ Klasifikasi Citra AI vs Human Anime")
st.write("Unggah gambar untuk mendeteksi apakah gambar merupakan **AI-Generated Anime**, **Human-Made Anime**, atau **Real Image**.")

# Pemetaan Indeks ke Nama Label
CLASS_MAP = {
    0: "🤖 AI-Generated Anime",
    1: "🎨 Human-Made Anime",
    2: "📷 Real Image / Non-Anime"
}

# 2. Fungsi Load Model (Di-cache agar cepat saat inferensi ulang)
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inisialisasi arsitektur EfficientNet-B0 dengan 3 output class
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 3)
    
    # Memuat bobot hasil pelatihan
    model.load_state_dict(torch.load('best_model_efficientnet_3class.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

# 3. Load Model
try:
    model, device = load_trained_model()
    st.sidebar.success("✅ Model 3-Class Berhasil Dimuat")
except Exception as e:
    st.error("❌ Gagal memuat file 'best_model_efficientnet_3class.pth'. Pastikan file berada di folder yang sama.")
    st.stop()

# 4. Pipeline Preprocessing Gambar (Sesuai tahap Validation/Testing)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Component Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar di sini...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    
    # Preprocess & Predict
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    pred_label = CLASS_MAP[predicted_idx.item()]
    
    # Display Hasil Prediksi
    st.markdown("---")
    st.subheader(f"Hasil Prediksi: **{pred_label}**")
    st.write(f"Tingkat Kepercayaan (Confidence): **{confidence.item()*100:.2f}%**")
    
    # Display Detail Probabilitas Setiap Kelas
    st.write("---")
    st.subheader("Detail Probabilitas:")
    for idx, prob in enumerate(probabilities):
        prob_percent = prob.item() * 100
        st.write(f"**{CLASS_MAP[idx]}**: {prob_percent:.2f}%")
        st.progress(float(prob))
