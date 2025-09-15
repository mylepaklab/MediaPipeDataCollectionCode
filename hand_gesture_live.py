import cv2
import mediapipe as mp
import numpy as np
import joblib  # for sklearn model
# or import tensorflow as tf if using TF model

# Load your trained model and label encoder
model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Extract landmarks from MediaPipe hand_landmarks object and flatten"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip image for natural selfie view
    image = cv2.flip(image, 1)
    # Convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks as features for model
        landmarks = extract_landmarks(hand_landmarks)

        # Reshape and predict gesture
        landmarks = landmarks.reshape(1, -1)  # 1 sample, n_features
        prediction = model.predict(landmarks)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display predicted label on the image
        cv2.putText(image, f'Gesture: {predicted_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
