import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your CNN model
model = load_model("asl_model.h5")

# Labels that match your training order
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "SPACE", "NOTHING"]

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 324, 324
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI for CNN
    img = cv2.resize(roi, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    # Draw box and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Detection (Image-Based)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
