# app.py
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CV CIFAR-10 Project", layout="centered")
st.title("CV CIFAR-10 Project")
st.write("Upload an image and get predictions from the trained CIFAR-10 CNN model.")

@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("cifar10_cnn_model.keras")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_cnn_model()

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = np.array(image.resize((32,32)))  # CIFAR-10 input size
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 32, 32, 3)

    prediction_probs = model.predict(img_array)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]
    st.success(f"Predicted Class: {class_names[predicted_class]}")

if st.checkbox("Show Confusion Matrix"):
    cm_path = "results/confusion_matrix.npy"
    if os.path.exists(cm_path):
        cm = np.load(cm_path)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix not found. Make sure 'results/confusion_matrix.npy' exists.")
