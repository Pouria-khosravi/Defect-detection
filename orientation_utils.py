# orientation_utils.py
"""
Orientation Utilities for Crack Detection
-----------------------------------------
This module provides helper functions to determine the orientation of
cracks from bounding boxes and to generate labels for visualization.

Usage:
- Used inside `video_crack_detection.py` to display orientation (Horizontal,
  Vertical, Diagonal) for detected cracks.
- Can also be reused in other YOLO-based detection pipelines that
  require orientation analysis.

Functions:
- get_orientation(bbox) → Returns "Horizontal", "Vertical", or "Diagonal"
  based on the geometry of the bounding box.
- get_label_with_orientation(cls, bbox, confidence, class_names, crack_class_name="crack")
  → Builds a text label for display, including orientation for cracks.
"""

import math


def get_orientation(bbox):
    """
    Determines the orientation of a crack based on the bounding box dimensions.

    Args:
        bbox (list or tuple): Bounding box coordinates [x, y, x2, y2].

    Returns:
        str: "Horizontal", "Vertical", or "Diagonal" depending on the angle.
    """
    (x, y, x2, y2) = bbox
    width = x2 - x
    height = y2 - y
    angle = math.degrees(math.atan2(height, width))

    if abs(angle) < 20:
        return "Horizontal"
    elif abs(angle) > 70:
        return "Vertical"
    else:
        return "Diagonal"


def get_label_with_orientation(cls, bbox, confidence, class_names, crack_class_name="crack"):
    """
    Generates a label with orientation for the 'crack' class,
    and only class name + confidence for others.

    Args:
        cls (int): Class index.
        bbox (list or tuple): Bounding box coordinates [x, y, x2, y2].
        confidence (float): Confidence score of the detection.
        class_names (list): List of class names.
        crack_class_name (str): Name of the crack class (default: "crack").

    Returns:
        str: Label text for display.
             Example: "crack (0.92), Horizontal" or "spall (0.85)"
    """
    class_name = class_names[cls] if cls < len(class_names) else "Unknown"

    if class_name.lower() == crack_class_name.lower():
        orientation = get_orientation(bbox)
        return f"{class_name} ({confidence:.2f}), {orientation}"
    else:
        return f"{class_name} ({confidence:.2f})"
