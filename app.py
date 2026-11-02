from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Detektor Anime AI", page_icon="🎨")

st.title("🎨 Klasifikasi Citra Anime 2D (Human vs AI)")
st.caption(
    "⚠️ **Ruang Lingkup Aplikasi:** Sistem ini dirancang khusus untuk membedakan ilustrasi 2D Anime buatan Manusia dan AI."
)


# Load Model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(
        torch.load("best_model_efficientnet.pth", map_location="cpu")
    )
    model.eval()
    return model


model = load_model()
class_names = ["AI-generated", "Human-made"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

uploaded_file = st.file_uploader(
    "Unggah Ilustrasi Anime (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits[0], dim=0)
        idx = torch.argmax(probs).item()
        confidence = probs[idx].item() * 100

    # Tampilkan Hasil Utama
    st.success(f"**Hasil Prediksi:** {class_names[idx]}")
    st.metric(label="Tingkat Keyakinan Model", value=f"{confidence:.2f}%")
