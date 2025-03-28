import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Load the YOLO model efficiently with caching
@st.cache_resource()
def load_model():
    model = YOLO(
        "/Users/saadiahlazim/Desktop/uthm_tech_asgmnt/runs/train6/weights/best.pt"
    )  # Ensure 'best.pt' is in the project directory
    return model


# Function to run inference
def run_inference(model, image):
    results = model(image)
    return results


# Streamlit UI
st.title("Car Detection with YOLOv5")
st.write("Upload an image to detect cars.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    results = run_inference(model, image_np)

    for result in results:
        img = result.plot()
        st.image(img, caption="Detected Cars", use_column_width=True)

        # Display detected car labels
        st.write("Detected Cars:")
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            class_name = model.names[int(class_id)]  # Get class label from model
            st.write(f"Class: {class_name}, Confidence: {conf:.2f}")

    st.success("Detection complete!")
