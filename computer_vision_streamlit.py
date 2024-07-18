import streamlit as st
import plotly.graph_objects as go
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(layout="wide")

@st.cache_resource
def load_model(model_type):
    return YOLO(model_type)

def process_image(image_bytes, model):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img)
    return results[0]

def plot_results(image, result, task):
    fig = go.Figure()

    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)

    fig.add_trace(go.Image(z=img_array))

    if task in ["det", "seg", "pose"]:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        names = result.names

        for box, cls in zip(boxes, classes):
            x0, y0, x1, y1 = box
            label = names[int(cls)]
            fig.add_shape(
                type="rect",
                x0=x0, y0=y0, x1=x1, y1=y1,
                line=dict(color="red", width=2),
            )
            fig.add_annotation(
                x=x0, y=y0,
                text=label,
                showarrow=False,
                bgcolor="red",
                font=dict(color="white")
            )

    if task == "pose":
        keypoints = result.keypoints.xy.cpu().numpy()
        for person_kps in keypoints:
            # Define the connections for a basic skeleton
            connections = [
                (5, 7), (7, 9), (6, 8), (8, 10),  # arms
                (5, 6), (5, 11), (6, 12),  # shoulders to hips
                (11, 13), (13, 15), (12, 14), (14, 16)  # legs
            ]
            
            for connection in connections:
                start, end = connection
                if person_kps[start][0] > 0 and person_kps[end][0] > 0:  # Check if keypoints are detected
                    fig.add_trace(go.Scatter(x=person_kps[[start, end], 0],
                                             y=person_kps[[start, end], 1],
                                             mode='lines',
                                             line=dict(color="lime", width=2)))
            
            # Plot keypoints
            fig.add_trace(go.Scatter(x=person_kps[:, 0], y=person_kps[:, 1],
                                     mode='markers',
                                     marker=dict(color="red", size=5)))

    if task == "seg":
        masks = result.masks.xy
        for i, mask in enumerate(masks):
            fig.add_trace(go.Scatter(x=mask[:, 0], y=mask[:, 1], 
                                     fill="toself", 
                                     opacity=0.5, 
                                     fillcolor=f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 0.5)',
                                     line_color=None,
                                     showlegend=False))

    fig.update_layout(
        height=600,
        width=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

def plot_classification_results(image, result):
    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)

    fig = go.Figure(go.Image(z=img_array))

    height, width = img_array.shape[:2]
    probs = result.probs
    top5_labels = [result.names[i] for i in probs.top5]
    top5_values = probs.top5conf.tolist()

    for i, (label, value) in enumerate(zip(top5_labels, top5_values)):
        y_pos = height * 0.1 + i * height * 0.05
        fig.add_annotation(
            x=width * 0.05,
            y=y_pos,
            text=f"{label}: {value:.2f}",
            showarrow=False,
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.7)",
            font=dict(color="black")
        )

    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

st.title("🖼️ YOLOv8 Computer Vision Demo")

task = st.sidebar.selectbox("Select Task", ["Object Detection", "Pose Estimation", "Segmentation", "Classification"])

if task == "Object Detection":
    model = load_model('yolov8n.pt')
    task_code = "det"
elif task == "Pose Estimation":
    model = load_model('yolov8n-pose.pt')
    task_code = "pose"
elif task == "Segmentation":
    model = load_model('yolov8n-seg.pt')
    task_code = "seg"
else:
    model = load_model('yolov8n-cls.pt')
    task_code = "cls"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    
    try:
        result = process_image(image_bytes, model)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_bytes)
        
        with col2:
            st.subheader(f"{task} Result")
            if task_code != "cls":
                fig = plot_results(image_bytes, result, task_code)
            else:
                fig = plot_classification_results(image_bytes, result)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display additional information
        if task_code != "cls":
            st.subheader("Detected Objects")
            for i, (cls, conf) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
                st.write(f"{i+1}. {result.names[int(cls.item())]} (Confidence: {conf.item():.2f})")
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        st.error("Please try uploading a different image.")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This demo showcases various computer vision tasks using YOLOv8 models. Upload an image and see the results!")