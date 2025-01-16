import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the model
model_path = '/Users/r1/Documents/Projets_python/IA/Cat_Dog_Binary_Classification/models/imageclassifier2.h5'
model = load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prediction function
def predict_image(image):
    """
    Takes an image, preprocesses it, and returns the model's prediction.
    """
    image = image.resize((128, 128))  # Resize to match the model's input
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return "Dog" if prediction > 0.5 else "Cat"

# Streamlit UI
st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or dog, and the model will predict which it is.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Predict and display the result
        label = predict_image(image)
        st.write(f"The model predicts: **{label}**")
    except Exception as e:
        st.error(f"Error processing the image: {e}")