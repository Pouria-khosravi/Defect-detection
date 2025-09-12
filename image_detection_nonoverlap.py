# image_detection_nonoverlap.py
"""
Image Crack Detection with YOLO (Non-Overlapping Bounding Boxes)
----------------------------------------------------------------
This script runs YOLO detection on a single image and removes
overlapping bounding boxes of the same class based on Intersection-over-Union (IoU).
Only the larger or more confident box is kept for each overlapping region.

⚠️ Note:
- Unlike `video_crack_detection.py`, this script does **NOT** provide
  crack orientation labeling (Horizontal, Vertical, Diagonal).
- It processes a **single image**, not a video.
- No keyboard controls are needed here (the image is displayed until a key is pressed).

Features:
- Loads custom classes from a text file.
- Runs YOLO detection on an input image.
- Removes overlapping boxes of the same class using IoU.
- Assigns distinct colors per class for easy visualization.
- Saves the processed output image after displaying it.

Dependencies:
- ultralytics
- opencv-python
- numpy

Usage:
1. Place your trained YOLO model file (e.g., best.pt) in the project directory.
2. Update the configuration paths:
   - image_path   = "path/to/image.jpg"
   - classes_file = "path/to/classes.txt"
   - model_path   = "path/to/best.pt"
3. Run the script:
   python image_detection_nonoverlap.py
"""

import cv2
import numpy as np
from ultralytics import YOLO


# -------------------------
# Utility Functions
# -------------------------
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Intersection
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_area(box):
    """Calculate the area of a bounding box."""
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def remove_overlapping_boxes(bboxes, classes, confidences, iou_threshold=0.3):
    """
    Remove overlapping boxes of the same class, keeping the larger or more confident one.

    Args:
        bboxes (ndarray): Array of bounding boxes [x1, y1, x2, y2].
        classes (ndarray): Array of class indices.
        confidences (ndarray): Array of confidence scores.
        iou_threshold (float): IoU threshold for overlap removal.

    Returns:
        tuple: Filtered (bboxes, classes, confidences).
    """
    if len(bboxes) == 0:
        return bboxes, classes, confidences

    keep_indices = []

    for i in range(len(bboxes)):
        should_keep = True
        for j in range(len(bboxes)):
            if i != j and classes[i] == classes[j]:  # same class
                iou = calculate_iou(bboxes[i], bboxes[j])
                if iou > iou_threshold:
                    area_i = calculate_area(bboxes[i])
                    area_j = calculate_area(bboxes[j])
                    if area_i < area_j:
                        should_keep = False
                        break
                    elif area_i == area_j and confidences[i] < confidences[j]:
                        should_keep = False
                        break
        if should_keep:
            keep_indices.append(i)

    return bboxes[keep_indices], classes[keep_indices], confidences[keep_indices]


# -------------------------
# Configuration (update these paths)
# -------------------------
image_path = "path/to/image.jpg"
classes_file = "path/to/classes.txt"
model_path = "path/to/best.pt"

# Define colors for classes (BGR format)
class_colors = [
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (0, 0, 255),   # Red
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (128, 0, 128), # Purple
    (255, 165, 0), # Orange
    (0, 128, 255), # Light Blue
    (255, 20, 147),# Deep Pink
    (0, 255, 127), # Spring Green
    (255, 69, 0),  # Red Orange
    (75, 0, 130),  # Indigo
    (255, 215, 0), # Gold
    (220, 20, 60), # Crimson
    (0, 206, 209), # Dark Turquoise
    (255, 105, 180),# Hot Pink
    (124, 252, 0), # Lawn Green
    (255, 140, 0), # Dark Orange
    (72, 61, 139)  # Dark Slate Blue
]


# -------------------------
# Main
# -------------------------
# Load class names
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLO model
model = YOLO(model_path)

# Load image
frame = cv2.imread(image_path)
if frame is None:
    print(f"Error: Unable to load image from {image_path}")
    exit()

# Resize for YOLO input
frame = cv2.resize(frame, (640, 640))

# Run detection
results = model(frame, device="mps", conf=0.4)
result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
classes = np.array(result.boxes.cls.cpu(), dtype="int")
confidences = np.array(result.boxes.conf.cpu())

# Remove overlapping boxes
bboxes, classes, confidences = remove_overlapping_boxes(bboxes, classes, confidences, iou_threshold=0.3)
print(f"Kept {len(bboxes)} detections after overlap removal")

# Draw detections
for cls, bbox, confidence in zip(classes, bboxes, confidences):
    (x, y, x2, y2) = bbox
    class_name = class_names[cls] if cls < len(class_names) else "Unknown"
    color = class_colors[cls % len(class_colors)]
    label = f"{class_name} ({confidence:.2f})"

    # Bounding box
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

    # Label with white background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    cv2.rectangle(frame, (x, y - h - 5), (x + w, y + 5), (255, 255, 255), -1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

# Show result
cv2.imshow("YOLO Detection Result", frame)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()

# Save result
cv2.imwrite("path/to/save/detected_image.jpg", frame)
