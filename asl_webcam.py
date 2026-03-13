import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (image-based)
model = load_model("asl_model.h5")

# Define class labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "SPACE", "NOTHING"]

# Open webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess the image for model
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict the sign
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    # Show prediction on screen
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
