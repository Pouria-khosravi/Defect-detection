

# ============================================
# Crack Detection & Target Assignment Script
# --------------------------------------------
# This script processes images of bridges to detect cracks (or other defects)
# using YOLO models and overlays target markers on the detected defects.
# Features include:
# 1. Dynamic YOLO model switching at runtime.
# 2. Adjustable detection confidence.
# 3. Consistent defect-to-target assignment using hashing.
# 4. EXIF preservation when saving images.
# 5. Export of target-defect mapping CSVs for Agisoft and approved defects.
# 6. Visual display with bounding boxes, class names, and target numbers.
# 7. Statistics tracking for total detections and approved defects.
# 
# Controls:
# - ESC: Exit
# - Right Arrow: Next image
# - Left Arrow: Previous image
# - H: Increase confidence (+0.05)
# - L: Decrease confidence (-0.05)
# - M: Switch to next YOLO model
# - Space: Toggle pause/auto-advance
# - S: Save image WITH targets (approve defects)
# - N: Save image WITHOUT targets (do not approve defects)
# 
# Notes:
# - Targets per defect class are cycled to ensure consistency.
# - Bounding boxes, labels, and targets are drawn on a copy of the original frame.
# - Supports auto-advance mode for batch processing.
# - Handles image clipping for targets at image edges.
# - Color coding is applied per class for easy visualization.
# - All paths to images, classes, weights, and save folders must be set correctly
#   in the variables below before running the script.
# ============================================
#structural_defect_annotator.py
import cv2
import os
import random
import piexif
import piexif.helper
from ultralytics import YOLO
import numpy as np
from PIL import Image
import hashlib
import csv

# Paths - MAKE SURE THESE PATHS ARE CORRECT FOR YOUR SYSTEM
image_folder = "/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/Bridge_Magenta"
targets_folder = "/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/targets"
save_folder_with_targets = "/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/jpeg-target"
save_folder_without_targets = "/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/jpeg-original"
os.makedirs(save_folder_with_targets, exist_ok=True)
os.makedirs(save_folder_without_targets, exist_ok=True)

# Load class names
# MAKE SURE THIS PATH IS CORRECT FOR YOUR SYSTEM
with open("/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# --- NEW FEATURE: Dynamic Model Switching Setup ---
weights_folder = "/Users/pouryakhosravi/PycharmProjects/crackdetection/crackdetection/.venv/bin/weights"
# Get all .pt files in the weights folder
model_files = [f for f in os.listdir(weights_folder) if f.lower().endswith('.pt')]
model_paths = [os.path.join(weights_folder, f) for f in model_files]

if not model_paths:
    print(f"Error: No .pt model files found in {weights_folder}. Please ensure your weights are there.")
    exit() # Exit if no models are found

# Sort models for consistent cycling (optional, but good practice)
model_paths.sort()

# Initial model setup
active_model_index = 0
initial_model_name_preferred = "best-concretenov5v1.pt" # Your initial default model name

# Try to find the preferred initial model, otherwise use the first one found
try:
    active_model_index = next(i for i, path in enumerate(model_paths) if initial_model_name_preferred in os.path.basename(path))
    print(f"Loading initial preferred model: {os.path.basename(model_paths[active_model_index])}")
except StopIteration:
    print(f"Warning: Initial preferred model '{initial_model_name_preferred}' not found in weights folder.")
    print(f"Loading the first available model: {os.path.basename(model_paths[0])}")
    active_model_index = 0 # Default to first if specific model not found

# Load the initial YOLO model
# CORRECTED: Removed 'device="mps"' from YOLO constructor
model = YOLO(model_paths[active_model_index])
# --- END NEW FEATURE SETUP ---

# Get image files
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if
               file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Load target images
target_files = [os.path.join(targets_folder, file) for file in os.listdir(targets_folder) if
                file.lower().endswith('.png')]
available_targets = [cv2.imread(target_file, cv2.IMREAD_UNCHANGED) for target_file in target_files]

# Target assignment configuration
TOTAL_TARGETS = 42  # Total unique target IDs available across all classes
targets_per_class = TOTAL_TARGETS // len(class_names) if len(class_names) > 0 else TOTAL_TARGETS
print(f"Targets per class: {targets_per_class}")

# Dictionary to store consistent target assignments for each defect
defect_target_map = {}
class_target_counters = {}  # Track next available target for each class for consistent assignment
target_to_defect_info = {}  # Store mapping for the main target_defect_mapping.csv export

# List to store approved defect class_id and target_id for the new CSV export
approved_defects_csv_data = []

# Color palette for different classes (BGR format)
class_colors = [
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 255, 0),  # Green
    (255, 0, 0),  # Blue
    (0, 0, 255),  # Red
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (255, 192, 203),  # Pink
    (0, 255, 127),  # Spring Green
    (255, 20, 147),  # Deep Pink
    (30, 144, 255),  # Dodger Blue
    (255, 215, 0),  # Gold
    (50, 205, 50),  # Lime Green
    (220, 20, 60),  # Crimson
]

# Statistics
approved_defects_count = {}
total_detections_count = {}


def get_defect_hash(bbox, class_id, image_path):
    """Create a consistent hash for similar defects based on position and class"""
    img_shape = cv2.imread(image_path).shape
    x, y, x2, y2 = bbox
    # Normalize coordinates to be less sensitive to minor pixel shifts
    center_x_norm = ((x + x2) // 2) / img_shape[1]
    center_y_norm = ((y + y2) // 2) / img_shape[0]

    # Use a low precision for normalization to group similar defects more aggressively
    hash_input = f"{class_id}_{round(center_x_norm, 2)}_{round(center_y_norm, 2)}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_target_for_defect(defect_hash, class_id, class_name):
    """Get consistent target for a defect, assigning one if not exists"""
    if defect_hash not in defect_target_map:
        # Calculate target range for this class
        # Target IDs start from 1, so adjust class_id (0-indexed)
        start_target = class_id * targets_per_class + 1
        end_target = start_target + targets_per_class - 1

        # Initialize counter for this class if not exists
        if class_id not in class_target_counters:
            class_target_counters[class_id] = start_target

        # Get next available target number for this class
        target_number = class_target_counters[class_id]

        # Check if we've exceeded the target range for this class
        if target_number > end_target:
            print(
                f"Warning: Exceeded target range for class '{class_name}' (max {targets_per_class} targets per class). Resetting to start of range."
            )
            target_number = start_target  # Wrap around to beginning of range

        # Assign target image (cycle through available target images for the visual representation)
        # Use modulo len(available_targets) to ensure we always pick a valid image
        target_img = available_targets[(target_number - 1) % len(available_targets)].copy()
        if target_img.shape[2] == 4:  # Convert RGBA to BGR if it has an alpha channel
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR)

        # Store the assignment in the map
        defect_target_map[defect_hash] = {
            'target_img': target_img,
            'target_number': target_number,
            'class_name': class_name,
            'class_id': class_id
        }

        # Store info for the main target_defect_mapping.csv export
        target_to_defect_info[target_number] = {
            'defect_hash': defect_hash,
            'class_name': class_name,
            'class_id': class_id,
            'target_range': f"{start_target}-{end_target}"
        }

        # Increment counter for next defect of this class
        class_target_counters[class_id] += 1

        print(f"Assigned Target_{target_number} to {class_name} defect (Range: {start_target}-{end_target})")

    return defect_target_map[defect_hash]['target_img'], defect_target_map[defect_hash]['target_number']


def get_class_color(class_id):
    """Get consistent color for each class"""
    return class_colors[class_id % len(class_colors)]


def process_image(image_path, confidence_threshold):
    """Process a single image with YOLO detection"""
    global model # Declare model as global to ensure we use the current loaded model
    original_frame = cv2.imread(image_path)
    if original_frame is None:
        print(f"Error: Could not read image {image_path}. Skipping.")
        return None, None, [], None

    # Print original image resolution
    print(f"Original image resolution: {original_frame.shape[1]}x{original_frame.shape[0]}")

    display_frame = original_frame.copy()

    exif_data = {}
    try:
        pil_image = Image.open(image_path)
        exif_data = piexif.load(pil_image.info.get("exif", b""))
    except Exception as e:
        print(f"Warning: Could not load EXIF data for {os.path.basename(image_path)}: {e}")
        # Initialize with a basic structure if loading fails
        exif_data = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

    # Pass the dynamic confidence_threshold to the model
    results = model(original_frame, device="mps", conf=confidence_threshold)  # Using "mps" for Apple Silicon GPUs
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")

    detections = []

    # Count total detections for statistics (excluding specific classes)
    for cls in classes:
        class_name = class_names[cls] if cls < len(class_names) else "Unknown"
        # Example of excluding certain classes from target assignment/display
        if class_name.lower() not in ["dining table", "cup"]:
            total_detections_count[class_name] = total_detections_count.get(class_name, 0) + 1

    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox
        class_name = class_names[cls] if cls < len(class_names) else "Unknown"

        if class_name.lower() in ["dining table", "cup"]:
            continue

        # Get color for this class
        color = get_class_color(cls)

        # Get consistent target for this defect and display it
        defect_hash = get_defect_hash(bbox, cls, image_path)
        target_img, target_number = get_target_for_defect(defect_hash, cls, class_name)

        # Add all relevant info to detections list for later use (e.g., saving approved defects)
        detections.append({
            'bbox': bbox,
            'class_id': cls,
            'class_name': class_name, # Ensure class_name is here for the approved_defects_csv_data later
            'color': color,
            'target_number': target_number  # Include target number here
        })

        # Make targets smaller for display (and for saving with targets)
        scale_factor = 0.08  # You can adjust this value (e.g., 0.05 for 5%, 0.1 for 10%)
        new_width = max(int(target_img.shape[1] * scale_factor), 8)  # Ensure min size is 8x8
        new_height = max(int(target_img.shape[0] * scale_factor), 8)
        target_img_resized = cv2.resize(target_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Place target at center of detected defect on display frame
        center_x, center_y = (x + x2) // 2, (y + y2) // 2
        start_x, start_y = center_x - new_width // 2, center_y - new_height // 2
        end_x, end_y = start_x + new_width, start_y + new_height

        # Ensure target fits within image bounds before placing
        if 0 <= start_x and end_x <= display_frame.shape[1] and 0 <= start_y and end_y <= display_frame.shape[0]:
            display_frame[start_y:end_y, start_x:end_x] = target_img_resized
        else:
            # If target goes out of bounds, try to clip it or place partially
            clipped_start_x = max(0, start_x)
            clipped_start_y = max(0, start_y)
            clipped_end_x = min(display_frame.shape[1], end_x)
            clipped_end_y = min(display_frame.shape[0], end_y)

            if clipped_end_x > clipped_start_x and clipped_end_y > clipped_start_y:
                # Recalculate target_img_resized for the clipped portion
                clipped_target_width = clipped_end_x - clipped_start_x
                clipped_target_height = clipped_end_y - clipped_start_y

                # Calculate the portion of the resized target image to use
                src_x_start = max(0, -start_x)
                src_y_start = max(0, -start_y)
                src_x_end = new_width - max(0, end_x - display_frame.shape[1])
                src_y_end = new_height - max(0, end_y - display_frame.shape[0])

                clipped_portion = target_img_resized[src_y_start:src_y_end, src_x_start:src_x_end]

                display_frame[clipped_start_y:clipped_end_y, clipped_start_x:clipped_end_x] = clipped_portion

        # Draw bounding box, label, and target number on display frame
        cv2.rectangle(display_frame, (x, y), (x2, y2), color, 2)
        # Position text slightly above the bounding box
        text_x = x
        text_y = y - 10
        # Ensure text is not cut off at the top
        if text_y < 20:  # Adjust if your font size makes this too small
            text_y = y + 30  # Place below if too high

        cv2.putText(display_frame, f"{class_name} (T{target_number})", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    # Reduced font scale
                    color, 2)  # Reduced thickness

    return original_frame, display_frame, detections, exif_data


def create_frame_with_targets(original_frame, detections, image_path):
    """Create a frame with targets placed on detections for saving (optimized for Agisoft)"""
    frame_with_targets = original_frame.copy()

    for detection in detections:
        bbox = detection['bbox']
        cls = detection['class_id']
        class_name = detection['class_name']  # Need class_name for get_target_for_defect

        (x, y, x2, y2) = bbox

        # Get consistent target for this defect (already assigned in process_image, but ensures it's retrieved)
        defect_hash = get_defect_hash(bbox, cls, image_path)
        target_img, target_number = get_target_for_defect(defect_hash, cls, class_name)  # Pass class_name

        # Use the same scale factor as for display
        scale_factor = 0.08
        new_width = max(int(target_img.shape[1] * scale_factor), 8)
        new_height = max(int(target_img.shape[0] * scale_factor), 8)
        target_img_resized = cv2.resize(target_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Place target at center of detected defect
        center_x, center_y = (x + x2) // 2, (y + y2) // 2
        start_x, start_y = center_x - new_width // 2, center_y - new_height // 2
        end_x, end_y = start_x + new_width, start_y + new_height

        # Ensure target fits within image bounds (same logic as in process_image)
        if 0 <= start_x and end_x <= frame_with_targets.shape[1] and 0 <= start_y and end_y <= frame_with_targets.shape[
            0]:
            frame_with_targets[start_y:end_y, start_x:end_x] = target_img_resized
        else:
            clipped_start_x = max(0, start_x)
            clipped_start_y = max(0, start_y)
            clipped_end_x = min(frame_with_targets.shape[1], end_x)
            clipped_end_y = min(frame_with_targets.shape[0], end_y)

            if clipped_end_x > clipped_start_x and clipped_end_y > clipped_start_y:
                src_x_start = max(0, -start_x)
                src_y_start = max(0, -start_y)
                src_x_end = new_width - max(0, end_x - frame_with_targets.shape[1])
                src_y_end = new_height - max(0, end_y - frame_with_targets.shape[0])

                clipped_portion = target_img_resized[src_y_start:src_y_end, src_x_start:src_x_end]

                frame_with_targets[clipped_start_y:clipped_end_y, clipped_start_x:clipped_end_x] = clipped_portion

    return frame_with_targets


def save_image(frame, exif_data, image_path, folder, suffix=""):
    """Save image with metadata"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_name = f"{base_name}{suffix}.jpg"
    save_path = os.path.join(folder, save_name)

    # Print resolution of the frame being saved
    print(f"Saving image '{save_name}' with resolution: {frame.shape[1]}x{frame.shape[0]}")

    pil_output = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    exif_bytes = None
    try:
        exif_bytes = piexif.dump(exif_data)
    except Exception as e:
        print(f"Warning: Could not dump EXIF data for {os.path.basename(image_path)}: {e}")
        # If EXIF dumping fails, proceed without it
        exif_data = {}

    if exif_bytes:
        pil_output.save(save_path, "jpeg", exif=exif_bytes, quality=95) # Added quality for good balance
    else:
        pil_output.save(save_path, "jpeg", quality=95) # Added quality for good balance
    return save_path


def export_target_mapping_csv():
    """Export target to defect mapping for Agisoft import"""
    csv_path = os.path.join(save_folder_with_targets, "target_defect_mapping.csv")

    # Changed 'Defect_Type' to 'Class_Name' for clarity
    fieldnames = ['Target_ID', 'Target_Name', 'Class_Name', 'Class_ID', 'Target_Range', 'Defect_Hash']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        # Sort by Target_ID for better readability
        for target_num in sorted(target_to_defect_info.keys()):
            info = target_to_defect_info[target_num]
            # Create meaningful defect name
            defect_name = f"{info['class_name']}_{target_num:03d}"  # e.g., "Crack_001"

            writer.writerow({
                'Target_ID': f"Target_{target_num}",  # e.g., "Target_1"
                'Target_Name': defect_name,
                'Class_Name': info['class_name'], # Mapped to the new column name
                'Class_ID': info['class_id'],
                'Target_Range': info['target_range'],
                'Defect_Hash': info['defect_hash']
            })

    print(f"✓ Target mapping exported to: {csv_path}")
    return csv_path


def export_approved_targets_csv(data):
    """Export class_id, class_name, and target_id for approved defects."""
    csv_path = os.path.join(save_folder_with_targets, "approved_defects_class_target_ids.csv")

    # Added 'class_name' to fieldnames
    fieldnames = ['class_id', 'class_name', 'target_id']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"✓ Approved defects (class_id, class_name, target_id) exported to: {csv_path}")
    return csv_path


def print_statistics():
    """Print current statistics"""
    print("\n" + "=" * 50)
    print("DETECTION STATISTICS")
    print("=" * 50)
    print("Total Detections Across All Processed Images:")
    if total_detections_count:
        for class_name, count in sorted(total_detections_count.items()):
            print(f"  {class_name}: {count}")
    else:
        print("  No detections yet.")

    print("\nApproved Defects (Count of defects in images where 'S' was pressed):")
    if approved_defects_count:
        for class_name, count in sorted(approved_defects_count.items()):
            print(f"  {class_name}: {count}")
    else:
        print("  None approved yet.")
    print("=" * 50)


# Main processing loop
current_index = 0
paused = False
auto_advance = True
confidence_threshold = 0.5  # Initial confidence score

print("Controls:")
print("ESC - Exit")
print("Right Arrow - Next image")
print("Left Arrow - Previous image")
print("H - Increase confidence score (+0.05)")
print("L - Decrease confidence score (-0.05)")
print("M - Switch to next YOLO model (will cause a brief pause)")
print("Space - Toggle pause/auto-advance")
print("S - Save current image WITH targets (approves defects and adds to approved CSV)")
print("N - Save current image WITHOUT targets and go to next (does not approve defects)")

while current_index < len(image_files):
    if current_index < 0:
        current_index = 0
    elif current_index >= len(image_files):
        break  # End of images

    image_path = image_files[current_index]
    print(f"\nProcessing image {current_index + 1}/{len(image_files)}: {os.path.basename(image_path)}")

    # Process current image with the current confidence threshold
    original_frame, display_frame, detections, exif_data = process_image(image_path, confidence_threshold)

    if original_frame is None:  # Skip if image loading failed
        current_index += 1
        continue

    # Add current confidence score display to the frame
    cv2.putText(display_frame, f"Conf: {confidence_threshold:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) # Green text, adjusts with image size

    # Add current model name display
    current_model_name = os.path.basename(model_paths[active_model_index])
    cv2.putText(display_frame, f"Model: {current_model_name}", (10, 70), # Positioned below confidence
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) # Yellow color for model name


    # Display image with bounding boxes
    cv2.imshow("YOLO Detection & Targets", display_frame)

    # Handle keyboard input
    key_raw = cv2.waitKey(2000 if auto_advance and not paused else 0)
    key_char = key_raw & 0xFF # Use & 0xFF for character keys and common special keys

    if key_raw == 27:  # ESC key (common raw value)
        print("Exiting...")
        break
    # Keeping arrow key checks for Left/Right in case they work for you
    elif key_raw == 65363 or key_char == 3:  # Right arrow (Raw code for Right on macOS, or common char code)
        current_index += 1
        auto_advance = False
    elif key_raw == 65361 or key_char == 2:  # Left arrow (Raw code for Left on macOS, or common char code)
        current_index -= 1
        auto_advance = False
    elif key_char == ord('h') or key_char == ord('H'): # H key to Increase confidence
        confidence_threshold = min(1.0, confidence_threshold + 0.05)
        print(f"Confidence increased to: {confidence_threshold:.2f}")
        auto_advance = False # Stay on current image until manual next
        paused = True      # Pause to show effect
        continue           # Re-run loop for the same image with new confidence
    elif key_char == ord('l') or key_char == ord('L'): # L key to Decrease confidence
        confidence_threshold = max(0.01, confidence_threshold - 0.05) # Minimum confidence 0.01
        print(f"Confidence decreased to: {confidence_threshold:.2f}")
        auto_advance = False # Stay on current image until manual next
        paused = True      # Pause to show effect
        continue           # Re-run loop for the same image with new confidence
    elif key_char == ord('m') or key_char == ord('M'): # M key to switch model
        print("\nSwitching model... This may cause a brief pause.")
        active_model_index = (active_model_index + 1) % len(model_paths)
        new_model_path = model_paths[active_model_index]
        # CORRECTED: Removed 'device="mps"' from YOLO constructor
        model = YOLO(new_model_path)
        print(f"Switched to model: {os.path.basename(new_model_path)}")
        auto_advance = False # Pause after model switch
        paused = True
        continue # Re-run loop for the same image with the new model
    elif key_char == 32:  # Space key
        paused = not paused
        auto_advance = not paused  # Space toggles between paused and auto-advance
        print(f"{'Paused' if paused else 'Resumed auto-advance'}")
    elif key_char == ord('s') or key_char == ord('S'):  # S key - save WITH targets (and approve defects)
        # Iterate through detections and count approved defects for statistics
        for detection in detections:
            class_name = detection['class_name']
            approved_defects_count[class_name] = approved_defects_count.get(class_name, 0) + 1
            # Capture class_id and target_id for the new CSV specific to approved defects
            approved_defects_csv_data.append({
                'class_id': detection['class_id'],
                'class_name': detection['class_name'], # Now also include class_name
                'target_id': detection['target_number']
            })

        # Create the frame with targets for saving
        frame_with_targets = create_frame_with_targets(original_frame, detections, image_path)

        # Save the image with targets
        save_path = save_image(frame_with_targets, exif_data, image_path, save_folder_with_targets, "_with_targets")
        print(f"✓ Saved WITH targets: {save_path}")
        print(f"✓ Approved {len(detections)} defects in this image.")

        print_statistics()  # Print updated statistics after approval
        current_index += 1  # Move to the next image after saving and approving
    elif key_char == ord('n') or key_char == ord('N'):  # N key - save WITHOUT targets (do not approve defects)
        # Save the original image without any targets drawn on it
        save_path = save_image(original_frame, exif_data, image_path, save_folder_without_targets, "_original")
        print(f"✓ Saved WITHOUT targets: {save_path}")
        current_index += 1  # Move to the next image after saving without targets
    elif key_raw == -1 and auto_advance:  # Timeout in auto-advance mode (no key pressed within waitKey duration)
        # Auto-save the original image without targets and continue to the next
        save_path = save_image(original_frame, exif_data, image_path, save_folder_without_targets, "_original")
        print(f"✓ Auto-saved WITHOUT targets: {save_path}")
        current_index += 1

cv2.destroyAllWindows()

# Final statistics and CSV exports after the loop finishes
print_statistics()

# Export the main target mapping CSV for Agisoft import (contains all assigned targets, not just approved ones)
if target_to_defect_info:
    csv_path_mapping = export_target_mapping_csv()
    print(f"\n✓ Import '{os.path.dirname(csv_path_mapping)}' into Agisoft to rename targets with defect information.")

# Export the new CSV specifically for approved defects (class_id, class_name, target_id)
if approved_defects_csv_data:
    csv_path_approved = export_approved_targets_csv(approved_defects_csv_data)
    print(f"✓ Check '{csv_path_approved}' for list of approved class_id, class_name and target_id pairs.")

print(f"\nProcessing complete!")
print(f"Total images processed: {len(image_files)}")
print(f"Total defects approved (by pressing 'S'): {sum(approved_defects_count.values())}")
print(f"Total unique defect types that were assigned targets: {len(defect_target_map)}")
print(f"Number of classes for which targets were generated: {len(class_target_counters)}")
