import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ✅ Load Trained Model
print("📂 Loading trained model...")
model = tf.keras.models.load_model("sign_language_cnn.h5")

# ✅ Load Label Encoder
data_dict = pickle.load(open('data_cnn.pickle', 'rb'))
labels = np.array(data_dict['labels'])
unique_labels = sorted(set(labels))  # Get unique labels (A-Z)

# ✅ Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# ✅ Open Webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ ERROR: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture image.")
        continue

    # ✅ Convert frame to RGB for Mediapipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max, y_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # ✅ Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # ✅ Resize to match model input (64x64)
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:  # Ensure valid region
                hand_img = cv2.resize(hand_img, (64, 64), interpolation=cv2.INTER_AREA)

                hand_img = (hand_img - np.mean(hand_img)) / np.std(hand_img)  # Standardization

                hand_img = np.expand_dims(hand_img, axis=0)  # Reshape to (1, 64, 64, 3)

                # ✅ Make prediction
                prediction = model.predict(hand_img)
                predicted_index = np.argmax(prediction)
                predicted_character = unique_labels[predicted_index]
                print(f"🛠️ Model Output: {prediction}")
                print(f"✅ Predicted Index: {predicted_index}, Predicted Character: {predicted_character}")

                # ✅ Display Prediction
                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            # ✅ Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ✅ Show real-time webcam feed
    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()