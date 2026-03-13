import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("asl_model.h5")

# Labels A-Z
labels = [chr(i) for i in range(65, 91)]

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks as a flat array (42 values: x & y of 21 points)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            landmarks = np.array(landmarks).reshape(1, -1)  # shape (1, 42)

            # Predict
            prediction = model.predict(landmarks)
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index] if predicted_index < len(labels) else "Unknown"

            # Show prediction
            cv2.putText(frame, f'Prediction: {predicted_label}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
