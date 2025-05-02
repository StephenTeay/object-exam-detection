import os
import streamlit as st
import cv2
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image

# Constants
LOG_INTERVAL = 300  # 5 minutes in seconds
LOG_FILE = "object_logs.txt"

# Initialize session state
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = datetime.now()
if 'processing' not in st.session_state:
    st.session_state.processing = False

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

def process_frame(frame):
    img = Image.fromarray(frame)
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))[0]
    return {
        "labels": [categories[label] for label in prediction["labels"]],
        "boxes": prediction["boxes"].detach().cpu().numpy(),
        "scores": prediction["scores"].detach().cpu().numpy()
    }

def log_predictions(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{timestamp}]\n")
        f.write("Label\t\tCount\tPercentage\n")
        f.write("-"*40 + "\n")
        for res in results:
            f.write(f"{res['Label']}\t\t{res['Count']}\t{res['Percentage']:.1f}%\n")

# Streamlit UI
st.title("Cloud Video Analyzer ☁️🎥")

# Video capture component
video_html = """
<div id="video-container">
  <video id="video" width="640" height="480" autoplay></video>
  <canvas id="canvas" style="display:none;"></canvas>
</div>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => console.error('Error accessing camera:', err));
}

function captureFrame() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.8);
}
</script>
"""
st.components.v1.html(video_html, height=500)

# Processing controls
if st.button("Start Processing") and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.last_log_time = datetime.now()
    label_stats = {}
    
    while st.session_state.processing:
        # Capture frame from JS component
        frame_data = st.components.v1.html(
            "<script>window.parent.postMessage(captureFrame(), '*');</script>",
            height=0,
            width=0
        )
        
        if frame_data:
            # Process frame
            frame = np.frombuffer(frame_data.split(",")[1].encode(), np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make prediction
            prediction = process_frame(frame)
            
            # Update stats
            for label, score in zip(prediction["labels"], prediction["scores"]):
                if label not in label_stats:
                    label_stats[label] = {"count": 0, "total_score": 0.0}
                label_stats[label]["count"] += 1
                label_stats[label]["total_score"] += score
            
            # Check logging interval
            if (datetime.now() - st.session_state.last_log_time).seconds >= LOG_INTERVAL:
                total_score = sum(v["total_score"] for v in label_stats.values())
                results = [{
                    "Label": label.title(),
                    "Count": data["count"],
                    "Percentage": (data["total_score"] / total_score * 100) if total_score > 0 else 0
                } for label, data in label_stats.items()]
                
                log_predictions(results)
                st.session_state.last_log_time = datetime.now()
                label_stats = {}  # Reset stats after logging

if st.button("Stop Processing"):
    st.session_state.processing = False

# Display logs
if st.checkbox("Show Logs"):
    try:
        with open(LOG_FILE, "r") as f:
            st.text(f.read())
    except FileNotFoundError:
        st.warning("No logs available yet")
