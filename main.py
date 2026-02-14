import cv2
import pickle
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import time

# Load model and setup MediaPipe
model = load_model('models/asl.h5')
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

labels = {i: chr(65 + i) for i in range(26)} # Quick way to map 0-25 to A-Z

# --- Logic Variables ---
prediction_history = []
buffer_size = 20
current_word = ""       
final_sentence = "" 
last_added_char = ""
last_hand_time = time.time()

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret: break
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        last_hand_time = time.time() # Update timer because a hand is present
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmark processing
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Prediction
            prediction = model.predict(np.asarray([data_aux]), verbose=0)
            predicted_index = np.argmax(prediction, axis=1)[0]
            
            if predicted_index < 26:
                char = labels[predicted_index]
                prediction_history.append(char)

        if len(prediction_history) > buffer_size:
            prediction_history.pop(0) # Keep buffer at fixed size
            
            # Find the most common character in the last X frames
            most_common = Counter(prediction_history).most_common(1)[0]
            
            # If the most common char appears in 90% of the buffer
            if most_common[1] > (buffer_size * 0.9):
                stable_char = most_common[0]
                
                # Only add if it's a new character (prevents "AAAAA")
                if stable_char != last_added_char:
                    current_word += stable_char
                    last_added_char = stable_char
    
    else:
        if time.time() - last_hand_time > 2.0 and current_word != "":
            final_sentence += current_word + " "
            current_word = ""
            last_added_char = ""
            prediction_history = []

    # UI Overlay
    cv2.rectangle(frame, (0, H-80), (W, H), (255, 255, 255), -1)
    cv2.putText(frame, f"Word: {current_word}", (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f"Sentence: {final_sentence}", (10, H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    cv2.imshow('ASL to Text', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        current_word = ""
        final_sentence = ""

cap.release()
cv2.destroyAllWindows()
