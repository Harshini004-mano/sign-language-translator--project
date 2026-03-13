import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_model.h5")

# Define labels (A-Z)
labels = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to 64x64
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0  # normalize to 0-1
    img = np.expand_dims(img, axis=0)  # shape (1, 64, 64, 3)

    # Predict
    prediction = model.predict(img)
    
    # Debug: see prediction values and shape
    # print("Prediction shape:", prediction.shape)
    # print("Prediction values:", prediction)
    
    predicted_index = np.argmax(prediction)

    # Safe indexing
    if predicted_index >= len(labels):
        predicted_label = "Unknown"
    else:
        predicted_label = labels[predicted_index]

    # Show predicted label on the frame
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the webcam frame
    cv2.imshow("ASL Translator", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
