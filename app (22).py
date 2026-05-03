
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AgriGuard", layout="wide")
st.title("🌱 AgriGuard - Smart Crop Disease Detector")
st.subheader("Helping Pakistani Farmers Detect Crop Diseases Early")
st.caption("**Husnian** | Big Data Analysis Course")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("agri_guard_model.h5")

model = load_model()

class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot"
]

st.write("### Upload a Photo of Crop Leaf")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Analyze Disease"):
        with st.spinner("Analyzing the leaf..."):
            # Preprocess
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100
            
            st.success(f"**Predicted Disease:** {predicted_class.replace('_', ' ')}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Simple remedy
            if "healthy" in predicted_class.lower():
                st.success("🌿 Plant is Healthy! Keep up the good work.")
            else:
                st.warning("🛠️ **Recommendation:** Consult local agriculture extension officer or use appropriate fungicide.")

st.sidebar.info("""
**AgriGuard Project**\n
- Trained on Potato & Corn diseases\n
- Validation Accuracy: ~95.5%\n
- Big Data Analysis Course Project
""")
