import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Load trained model
model = load_model("asl_model.h5")

# Define labels (must match your model outputs)
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'
]

# Create deque to store last few predictions
smooth_predictions = deque(maxlen=7)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (Region of Interest)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess image for model
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    # Add to deque
    smooth_predictions.append(predicted_label)

    # If last few predictions are same → stable result
    if len(smooth_predictions) == smooth_predictions.maxlen:
        most_common = max(set(smooth_predictions), key=smooth_predictions.count)
    else:
        most_common = predicted_label

    # Display prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {most_common}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
