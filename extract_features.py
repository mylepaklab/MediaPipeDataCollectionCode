import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Prepare data storage
data = []
label = "thumbs_up"  # 🔁 Change this to the gesture name you're collecting

# Store the path of the index finger tip
path_points = []

print("[INFO] Starting capture. Press 's' to save a frame, 'r' to reset path, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract x, y, z of each of the 21 landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Track the index finger tip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            path_points.append((cx, cy))

            # Draw path
            for i in range(1, len(path_points)):
                cv2.line(frame, path_points[i - 1], path_points[i], (0, 255, 255), 2)
                cv2.circle(frame, path_points[i], 2, (255, 0, 255), -1)

            # Save when 's' key is pressed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                data.append(landmarks + [label])
                print(f"[INFO] Saved a frame with label: {label}")

    else:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand Gesture Path Tracker", frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        path_points.clear()
        print("[INFO] Path reset.")

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()

# Save to CSV
df = pd.DataFrame(data)
csv_file = f"{label}_landmarks.csv"
df.to_csv(csv_file, index=False, header=False)
print(f"[INFO] Saved data to {csv_file}")
