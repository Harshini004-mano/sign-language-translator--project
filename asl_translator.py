import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# Load your trained model
model = load_model("asl_model.h5")

# Labels (change if you trained fewer)
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

sentence = ""
last_pred = ""
last_time = time.time()

cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict sign
            prediction = model.predict(landmarks)
            predicted_index = np.argmax(prediction)
            predicted_label = labels[predicted_index]

            # Update text every 1.5s to reduce flickering
            if predicted_label != last_pred:
                if time.time() - last_time > 1.5:
                    sentence += predicted_label
                    last_time = time.time()
                    last_pred = predicted_label

            cv2.putText(frame, f"Sign: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Text: {sentence}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Translator (Landmark-based)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
