import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from huggingface_hub import hf_hub_download

model = load_model(hf_hub_download(
    repo_id="kaungkhantcoder/FaceMaskDetection",
    filename="mask_detection_model.h5"
))

print(f"Model downloaded to: {model}")

# Load the face mask detection model
# model = load_model("kaungkhantcoder/FaceMaskDetection")

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("Face Mask Detection (External Webcam)")

# Dropdown to select webcam index
camera_index = st.selectbox("Select Camera Index:", [0, 1])

# Start and Stop button
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

# Placeholder for the video feed
video_placeholder = st.empty()

if start_button:
    cap = cv2.VideoCapture(camera_index)  # Open selected camera

    if not cap.isOpened():
      st.error(f"⚠️ Cannot open webcam at index {camera_index}. Try a different index.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Could not access the camera. Try a different index.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if face_cascade.empty():
            st.error("⚠️ Face detection model not found! Check OpenCV installation.")

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]  # Crop face
            face = cv2.resize(face, (150, 150))  # Resize to model input size
            face = img_to_array(face) / 255.0  # Normalize
            face = np.expand_dims(face, axis=0)

            # Predict mask or no mask
            prediction = model.predict(face)[0][0]
            label = "Mask" if prediction < 0.5 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_column_width=True)

        # Stop the webcam when user clicks "Stop Camera"
        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
