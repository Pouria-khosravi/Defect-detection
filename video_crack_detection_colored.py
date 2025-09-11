# video_crack_detection_colored.py
"""
Video Crack Detection with YOLO (Class-Colored Bounding Boxes)
--------------------------------------------------------------
This script applies a YOLO model to detect cracks and other defects
in a video file. It assigns each class a distinct bounding box color
for easier visualization, displays real-time FPS, and supports pause/resume.

 Note:
Unlike `video_crack_detection.py`, this version does **NOT** provide
crack orientation labeling (Horizontal, Vertical, Diagonal). It only
shows the class name and confidence score.

Features:
- Loads custom classes from a text file.
- Runs YOLO detection on each frame of a video.
- Assigns a unique color to each detected class for easy distinction.
- Displays bounding boxes, class labels, and confidence scores.
- Supports pausing/resuming the video with the spacebar.
- Displays FPS for performance monitoring.

Controls:
- ESC      → Exit the program
- Spacebar → Pause/Resume video playback

Dependencies:
- ultralytics
- opencv-python
- numpy

Usage:
1. Place your trained YOLO model file (e.g., best.pt) in the project directory.
2. Update the configuration paths:
   - CLASSES_FILE = "classes.txt"
   - VIDEO_PATH   = "crack-video3.mp4"
   - MODEL_PATH   = "best-concretenov10v1.pt"
3. Run the script:
   python video_crack_detection_colored.py
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO

# -------------------------
# Configuration
# -------------------------
CLASSES_FILE = "classes.txt"            # Path to class names file
VIDEO_PATH = "crack-video3.mp4"         # Path to input video
MODEL_PATH = "best-concretenov10v1.pt"  # Path to trained YOLO model

# Class-specific colors (BGR format)
class_colors = [
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 128, 255),   # Light Blue
    (128, 128, 128), # Gray
    (255, 192, 203), # Pink
    (0, 128, 0),     # Dark Green
    (128, 0, 0),     # Dark Blue
    (0, 0, 128),     # Dark Red
    (255, 255, 255)  # White
]

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
        results = model(frame, device="mps")

        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        confidences = np.array(result.boxes.conf.cpu())

        # Draw bounding boxes and labels
        for cls, bbox, confidence in zip(classes, bboxes, confidences):
            (x, y, x2, y2) = bbox
            class_name = class_names[cls] if cls < len(class_names) else "Unknown"

            # Get color for this class
            color = class_colors[cls % len(class_colors)]

            # Label with class name and confidence (no orientation info)
            label = f"{class_name} ({confidence:.2f})"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            # Background for label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), (255, 255, 255), -1)

            # Put label text in black
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    # FPS display
    end_time = time.time()
    fps = 1 / (end_time - start_time) if (end_time - start_time) != 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    start_time = end_time

    cv2.imshow("Crack Detection (Colored)", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    if key == 32:  # Spacebar to pause/resume
        paused = not paused

cap.release()
cv2.destroyAllWindows()
