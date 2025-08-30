import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load your trained model and label encoder
model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("gesture_labels.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Start video capture
cap = cv2.VideoCapture(0)

def extract_hand_landmarks(hand_landmarks):
    """
    Extract (x, y, z) coordinates from hand landmarks into a flat list.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return landmarks

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for natural (mirror) view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark features
            features = extract_hand_landmarks(hand_landmarks)

            # Convert to numpy array and reshape for model input
            features_np = np.array(features).reshape(1, -1)

            # Predict gesture
            pred = model.predict(features_np)
            gesture = label_encoder.inverse_transform(pred)[0]

            # Display predicted gesture on frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show the frame
    cv2.imshow("Real-time Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
