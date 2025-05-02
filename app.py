import os
import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from PIL import Image

# Constants
LOG_INTERVAL = 300  # 5 minutes in seconds
LOG_FILE = "object_logs.txt"

# Initialize session state
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = time.time()
if 'cam_running' not in st.session_state:
    st.session_state.cam_running = False

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

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor,
        boxes=prediction["boxes"],
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]],
        width=2
    )
    return img_with_bboxes.numpy().transpose(1, 2, 0)

def log_predictions(results):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n[{timestamp}]\n" + \
                   "Label\t\tCount\tPercentage\n" + \
                   "\n".join([f"{res['Label']}\t\t{res['Count']}\t{res['Percentage']}" 
                        for res in results])
        
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n\n")
            
        st.success(f"Logged at {timestamp}")
    except Exception as e:
        st.error(f"Logging error: {str(e)}")

# Streamlit UI
st.title("Real-Time Object Detector 📸")
start_btn = st.button("Start Webcam" if not st.session_state.cam_running else "Stop Webcam")

if start_btn:
    st.session_state.cam_running = not st.session_state.cam_running

if st.session_state.cam_running:
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    table_placeholder = st.empty()
    log_placeholder = st.empty()

    try:
        while st.session_state.cam_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Make prediction
            prediction = make_prediction(img)
            img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)
            
            # Update display
            frame_placeholder.image(img_with_bbox, channels="RGB")

            # Calculate stats
            label_stats = {}
            for label, score in zip(prediction["labels"], prediction["scores"]):
                if label not in label_stats:
                    label_stats[label] = {"count": 0, "total_score": 0.0}
                label_stats[label]["count"] += 1
                label_stats[label]["total_score"] += score.item()

            # Prepare results
            total_score = sum(v["total_score"] for v in label_stats.values())
            results = []
            for label, data in label_stats.items():
                percentage = (data["total_score"] / total_score * 100) if total_score > 0 else 0
                results.append({
                    "Label": label.title(),
                    "Count": data["count"],
                    "Percentage": f"{percentage:.1f}%"
                })

            # Update table
            df = pd.DataFrame(results)
            table_placeholder.table(df)

            # Check logging interval
            current_time = time.time()
            if current_time - st.session_state.last_log_time >= LOG_INTERVAL:
                log_predictions(results)
                st.session_state.last_log_time = current_time

            time.sleep(0.1)  # Control frame rate

    finally:
        cap.release()
        st.session_state.cam_running = False
        st.experimental_rerun()

# Show log file
if os.path.exists(LOG_FILE):
    st.subheader("Prediction History")
    with open(LOG_FILE, "r") as f:
        st.text(f.read())
