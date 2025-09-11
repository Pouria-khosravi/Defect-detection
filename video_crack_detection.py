# video_crack_detection.py
"""
Video Crack Detection with YOLO
--------------------------------
This script uses a trained YOLO model to detect cracks and other defects
in a video stream. It supports pause/resume, displays FPS, and integrates
orientation labeling for cracks.

Dependencies:
- ultralytics
- opencv-python
- numpy
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from orientation_utils import get_label_with_orientation

# -------------------------
# Configuration
# -------------------------
CLASSES_FILE = "classes.txt"        # Path to class names file
VIDEO_PATH = "input_video.mkv"      # Path to input video
MODEL_PATH = "best-concretenov10v1.pt"  # Path to trained YOLO model

# -------------------------
# Load resources
# -------------------------
with open(CLASSES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(VIDEO_PATH)
model = YOLO(MODEL_PATH)

paused = False
current_frame = None
start_time = time.time()

# -------------------------
# Main loop
# -------------------------
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = frame

    frame = current_frame

    # Run detection only when not paused
    if not paused:
        # 'device' controls where the model runs:
        # - "cpu"  → run on CPU (works everywhere, but slower)
        # - "cuda" → run on NVIDIA GPU (fast, requires CUDA installed)
        # - "mps"  → run on Apple Silicon (M1/M2/M3) using Metal Performance Shaders
        # Change this depending on your hardware
        results = model(frame, device="mps")

        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu())

        # Draw bounding boxes and labels
        for cls, bbox, confidence in zip(classes, bboxes, confidences):
            (x, y, x2, y2) = bbox
            class_name = class_names[cls] if cls < len(class_names) else "Unknown"

            if class_name.lower() == "crack":
                label = get_label_with_orientation(cls, bbox, confidence, class_names)
            else:
                label = f"{class_name} ({confidence:.2f})"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)

            # Background for label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), (255, 255, 255), -1)

            # Put label text
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)

    # FPS display
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) != 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    start_time = end_time

    cv2.imshow("Crack Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    if key == 32:  # Spacebar to pause/resume
        paused = not paused

cap.release()
cv2.destroyAllWindows()
