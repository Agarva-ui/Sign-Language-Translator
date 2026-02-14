import os
import mediapipe as mp
import cv2

DATA_DIR = 'data'
SAVE_DIR = "hand_images"
os.makedirs(SAVE_DIR, exist_ok=True)
dirs= ['24','25']

for dir_ in dirs:
    class_dir = os.path.join(SAVE_DIR, dir_)
    os.makedirs(class_dir, exist_ok=True)

    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = mp.solutions.hands.Hands().process(img_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = img.shape

            for hand_landmarks in results.multi_hand_landmarks:
                x_list = []
                y_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

                # Bounding box
                x_min, x_max = max(0, min(x_list) - 80), min(w, max(x_list) + 40)
                y_min, y_max = max(0, min(y_list) - 80), min(h, max(y_list) + 40)

                print(f"Cropping hand from ({x_min}, {y_min}) to ({x_max}, {y_max})")
                break
                hand_crop = img[y_min:y_max, x_min:x_max]

                save_path = os.path.join(class_dir, img_path)
                cv2.imwrite(save_path, hand_crop)
        break
    break
