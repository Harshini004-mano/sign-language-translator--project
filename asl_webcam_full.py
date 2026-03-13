# asl_webcam_debug.py
# Robust debugging / inference script for your image-based ASL model.
# Save as file and run from Anaconda Prompt with conda activate asl_env

import cv2
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
import os

# -------- CONFIG ----------
MODEL_FILE = "asl_model.h5"   # your model file (must be in same folder)
ROI = (100, 100, 324, 324)    # x1, y1, x2, y2 - adjust if needed
IMG_SIZE = (64, 64)           # model input size
SMOOTH_LEN = 7                # number of frames to smooth over
SAVE_DIR = "debug_saved_rois" # folder to save ROI snapshots (for debugging)
# --------------------------

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading model:", MODEL_FILE)
try:
    model = load_model(MODEL_FILE)
except Exception as e:
    print("ERROR loading model:", e)
    raise SystemExit(1)

# Warm-up (important to avoid long first-frame delay)
dummy = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
print("Warming up model (one dummy predict)...")
model.predict(dummy, verbose=0)
print("Warm-up done.")

# Report model output shape
output_shape = model.output_shape
print("Model output shape:", output_shape)

# Prepare labels. Model output length = output_shape[-1]
# Default label ordering: A-Z then SPACE, DELETE, NOTHING (29 classes)
default_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DELETE", "NOTHING"]
num_outputs = output_shape[-1]

if num_outputs == len(default_labels):
    labels = default_labels
elif num_outputs > len(default_labels):
    # extend with generic names
    labels = default_labels + [f"CLASS_{i}" for i in range(len(default_labels), num_outputs)]
else:
    # fewer outputs: cut default list
    labels = default_labels[:num_outputs]

print("Using labels (count={}):".format(len(labels)))
print(labels)

# Smoothing buffer
smooth_buffer = deque(maxlen=SMOOTH_LEN)

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Make sure no other app is using it.")
    raise SystemExit(1)

print("Webcam opened. Press 'q' to quit.")
print("Press 's' to save ROI snapshot (for debugging).")
print("Make sure your hand fills the green box. Adjust ROI in the script if needed.")

last_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break
        frame = cv2.flip(frame, 1)  # mirror image for natural interaction

        x1, y1, x2, y2 = ROI
        # Validate ROI within frame
        h, w = frame.shape[:2]
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        roi = frame[y1c:y2c, x1c:x2c]

        # Show ROI preview in small window for debugging
        debug_roi = cv2.resize(roi, (200, 200)) if roi.size else np.zeros((200,200,3), dtype=np.uint8)

        # Preprocess the ROI for the CNN model
        try:
            img = cv2.resize(roi, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure color order
            img = img.astype("float32") / 255.0
            img_input = np.expand_dims(img, axis=0)  # shape (1, H, W, 3)
        except Exception as e:
            # If resizing fails, skip prediction
            print("Preprocessing error:", e)
            img_input = None

        prediction = None
        if img_input is not None:
            # Model predict (quiet)
            prediction = model.predict(img_input, verbose=0)
            # prediction is shaped (1, N)
            pred_probs = prediction.flatten()
            pred_idx = int(np.argmax(pred_probs))
            pred_conf = float(pred_probs[pred_idx])
            pred_label = labels[pred_idx] if pred_idx < len(labels) else f"CLASS_{pred_idx}"
        else:
            pred_label = "NO_INPUT"
            pred_conf = 0.0
            pred_probs = None
            pred_idx = None

        # Add label to smoothing buffer
        smooth_buffer.append(pred_label)
        if len(smooth_buffer) == smooth_buffer.maxlen:
            # majority vote
            most_common = max(set(smooth_buffer), key=smooth_buffer.count)
        else:
            most_common = pred_label

        # Draw ROI box and predicted text
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        text = f"{most_common}  ({pred_conf*100:.1f}%)"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        # Show small ROI preview + info
        cv2.putText(debug_roi, "ROI preview", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # Compose combined display
        combined = cv2.hconcat([cv2.resize(frame, (640, 480)), cv2.resize(debug_roi, (200, 480))])
        cv2.imshow("ASL Debugger - main | ROI", combined)

        # Print prediction vector occasionally for debug (every 30 frames)
        frame_count += 1
        if frame_count % 30 == 0 and pred_probs is not None:
            # show top 5 predictions
            top5_idx = np.argsort(-pred_probs)[:5]
            top5 = [(labels[i], float(pred_probs[i])) for i in top5_idx]
            print("Top5:", top5)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # Save the current ROI image for debugging/retraining
            timestamp = int(time.time())
            fname = os.path.join(SAVE_DIR, f"roi_{timestamp}.jpg")
            cv2.imwrite(fname, roi)
            print("Saved ROI to", fname)

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cleaned up. Exiting.")
