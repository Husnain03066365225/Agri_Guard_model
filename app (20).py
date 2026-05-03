
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="AgriGuard", layout="wide")
st.title("🌱 AgriGuard - Smart Crop Disease Detection")
st.subheader("Early Detection System for Pakistani Farmers")
st.caption("**Husnian** | Big Data Analysis Course")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("agri_guard_model.h5")

model = load_model()

# Class names (update according to your classes)
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight", 
    "Potato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot"
]

# Remedies (you can expand this)
remedies = {
    "Potato___Early_blight": "Use fungicide (Mancozeb). Remove infected leaves. Improve air circulation.",
    "Potato___Late_blight": "Use Ridomil Gold or Curzate. Destroy infected plants immediately.",
    "Potato___healthy": "Plant is healthy. Continue good farming practices.",
    "Corn_(maize)___Common_rust_": "Apply fungicide. Use resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use crop rotation and resistant seeds.",
    "Corn_(maize)___healthy": "Plant is healthy.",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": "Apply appropriate fungicide and improve field drainage."
}

st.write("### Upload Crop Leaf Image")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Predict Disease"):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            st.success(f"**Prediction:** {predicted_class.replace('_', ' ')}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Show Remedy
            remedy = remedies.get(predicted_class, "Consult local agriculture expert.")
            st.info(f"**Recommended Action:** {remedy}")

st.sidebar.header("About Project")
st.sidebar.info("""
This project helps farmers detect crop diseases early using AI.
- Trained on Potato & Corn diseases
- Accuracy: ~95.5%
- Developed as Big Data Analysis Course Project
""")
