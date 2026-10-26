import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Uji Coba Detektor Anime AI")

# 1. Load Model ringkas
@st.cache_resource
def load_model():
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load('best_model_efficientnet.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
# Urutan label sesuai abjad nama folder (AI = 0, Human = 1)
class_names = ['AI-generated', 'Human-made']

# 2. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Upload & Prediksi
uploaded_file = st.file_uploader("Pilih gambar anime", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=300)
    
    # Proses prediksi
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs[0], dim=0)
        idx = torch.argmax(probs).item()
    
    # Tampilkan hasil sederhana
    st.write(f"**Prediksi:** {class_names[idx]}")
    st.write(f"**Keyakinan:** {probs[idx].item() * 100:.2f}%")