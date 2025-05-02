import os
import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Constants
LOG_FILE = "object_logs.txt"

# Model setup
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def log_prediction(results):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n[{timestamp}]\n")
            for res in results:
                f.write(f"{res['Label']} - {res['Count']} ({res['Percentage']})\n")
        st.success("Prediction logged successfully!")
    except Exception as e:
        st.error(f"Logging failed: {str(e)}")

# Streamlit UI
st.title("Cloud Object Detector ☁️")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Process image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        prediction = make_prediction(image)
    
    # Process results
    label_stats = {}
    for label, score in zip(prediction["labels"], prediction["scores"]):
        if label not in label_stats:
            label_stats[label] = {"count": 0, "total_score": 0.0}
        label_stats[label]["count"] += 1
        label_stats[label]["total_score"] += score.item()
    
    # Prepare display data
    total_score = sum(v["total_score"] for v in label_stats.values())
    results = []
    for label, data in label_stats.items():
        percentage = (data["total_score"] / total_score * 100) if total_score > 0 else 0
        results.append({
            "Label": label.title(),
            "Count": data["count"],
            "Percentage": f"{percentage:.1f}%"
        })
    
    # Show results
    st.subheader("Detection Results")
    df = pd.DataFrame(results)
    st.table(df)
    
    # Log prediction
    log_prediction(results)

# Show historical logs
if st.button("Show Prediction History"):
    try:
        with open(LOG_FILE, "r") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.warning("No predictions logged yet")
