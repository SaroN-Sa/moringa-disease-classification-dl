# =========================================
# MORINGA LEAF DISEASE CLASSIFIER WEB APP
# =========================================

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------
st.set_page_config(
    page_title="Moringa Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# -----------------------------------------
# DOWNLOAD MODEL (FROM GOOGLE DRIVE)
# -----------------------------------------
MODEL_PATH = "best_cnn_model.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait"):
        url = "https://drive.google.com/file/d/11V_24EhyY4V2FoONBmcU3OJQ4oof3PZA/view?usp=sharing"
        gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# -----------------------------------------
# CLASS LABELS
# -----------------------------------------
class_labels = [
    "Bacterial Leaf Spot",
    "Cercospora Leaf Spot",
    "Healthy",
    "Yellow"
]

# -----------------------------------------
# TITLE
# -----------------------------------------
st.title("🌿 Moringa Leaf Disease Classifier")

st.markdown("""
Upload a moringa leaf image and get instant disease prediction.

### Supported Classes:
- Healthy
- Yellow
- Bacterial Leaf Spot
- Cercospora Leaf Spot
""")

# -----------------------------------------
# IMAGE UPLOAD
# -----------------------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------
# PREPROCESS FUNCTION
# -----------------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    return np.expand_dims(img, axis=0)

# -----------------------------------------
# PREDICTION
# -----------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)

    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_labels[predicted_index]

    # -----------------------------------------
    # RESULT
    # -----------------------------------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    # -----------------------------------------
    # PROBABILITY CHART
    # -----------------------------------------
    st.subheader("Prediction Probabilities")

    prob_dict = {
        class_labels[i]: float(prediction[0][i])
        for i in range(len(class_labels))
    }

    st.bar_chart(prob_dict)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.caption("Developed using CNN for Moringa Disease Detection")
