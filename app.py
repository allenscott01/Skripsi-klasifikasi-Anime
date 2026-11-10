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
st.set_page_config(page_title="Detektor Forensik AI-Anime (IG)", page_icon="🛡️", layout="wide")

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
# 2. INFERENSI & INTEGRATED GRADIENTS (IG)
# ==========================================
def predict_and_ig_fixed(pil_img, n_steps=25):
    img_resized = pil_img.resize((224, 224))
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # 1. Prediction Pass
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    # 2. Gaussian Blur Baseline
    img_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    blurred_np = cv2.GaussianBlur(img_np, (21, 21), 0)
    baseline = (
        torch.tensor(blurred_np.transpose(2, 0, 1), dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # 3. IG Pass DINAMIS mengikuti kelas hasil prediksi (pred_idx)
    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=DEVICE)
    scaled_inputs = torch.cat(
        [baseline + alpha * (input_tensor - baseline) for alpha in alphas],
        dim=0,
    )
    scaled_inputs.requires_grad_(True)

    scaled_outputs = model(scaled_inputs)
    target_outputs = scaled_outputs[:, pred_idx]

    model.zero_grad()
    grads = torch.autograd.grad(
        outputs=target_outputs.sum(), inputs=scaled_inputs
    )[0]

    avg_grads = torch.mean(grads, dim=0, keepdim=True)
    integrated_grads = (input_tensor - baseline) * avg_grads

    # 4. Ambil Atribusi Positif untuk Kelas yang Terdeteksi
    ig_np = integrated_grads.squeeze(0).cpu().detach().numpy()
    ig_net = np.sum(ig_np, axis=0)  # Agregasi RGB
    pos_attr = np.maximum(0, ig_net)  # Hanya bukti positif pendorong keputusan

    # Normalisasi terbatas (mencegah magnifikasi noise 0.000%)
    max_val = np.max(pos_attr)
    if max_val > 1e-5:
        attr_norm = pos_attr / max_val
    else:
        attr_norm = np.zeros_like(pos_attr)

    # 5. Pewarnaan Dinamis Berdasarkan Kelas Terdeteksi
    heatmap_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

    if pred_idx == 0:  # AI-Anime -> Merah (🔴)
        heatmap_rgb[:, :, 0] = np.uint8(255 * attr_norm)
    elif pred_idx == 1:  # Human-Anime -> Biru (🔵)
        heatmap_rgb[:, :, 2] = np.uint8(255 * attr_norm)
    else:  # Real-Image (OOD) -> Hijau (🟢)
        heatmap_rgb[:, :, 1] = np.uint8(255 * attr_norm)

    # Overlay dengan kontras yang seimbang
    orig_array = np.array(img_resized)
    overlay = cv2.addWeighted(orig_array, 0.6, heatmap_rgb, 0.4, 0)

    return img_resized, overlay, pred_idx, probs
# ==========================================
# 3. ANTARMUKA DASHBOARD
# ==========================================
st.title("🛡️ Detektor Forensik Citra AI-Anime (XAI Integrated Gradients)")
st.caption(f"Perangkat Komputasi Aktif: {DEVICE.type.upper()} | XAI: Integrated Gradients (Pixel-Level Attribution)")

uploaded_file = st.file_uploader("Unggah gambar uji (PNG, JPG, JPEG, WEBP):", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Menganalisis piksel & menghitung Diverging IG..."):
        orig_img, ig_overlay, pred_idx, probs = predict_and_ig_diverging(img)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(
            orig_img, caption="Gambar Input Original", use_container_width=True
        )

    with col2:
        st.image(
            ig_overlay,
            caption="Diverging Attribution Map (Red vs Blue)",
            use_container_width=True,
        )
        # Legenda Indikator Warna
        st.markdown(
            """
        **Keterangan Warna Heatmap:**
        - 🔴 **Piksel Merah:** Jejak Fitur AI (Residu noise VAE / Over-smoothing).
        - 🔵 **Piksel Biru:** Jejak Fitur Human (Konsistensi stroke line-art / cell-shading).
        """
        )

    with col3:
        st.subheader("Hasil Analisis Forensik")
        st.metric(
            "Kategori Terdeteksi",
            CLASS_NAMES[pred_idx],
            f"{probs[pred_idx]*100:.2f}% Confidence",
        )
        st.divider()

        for i, name in enumerate(CLASS_NAMES):
            st.write(f"**{name}**: {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))
