import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("asl_model.h5")

# Define class labels (must match your model's output)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Only 26 letters

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (hand area)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess image
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)

    # Safety check: avoid index errors
    if predicted_index >= len(labels):
        predicted_label = "Unknown"
    else:
        predicted_label = labels[predicted_index]

    # Display results
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
