import os
import pickle

import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './words_data'

data = []
labels = []
for word_dir in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, word_dir)):
        continue
        
    for img_path in os.listdir(os.path.join(DATA_DIR, word_dir)):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, word_dir, img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  # Only process if exactly one hand is detected
            hand_landmarks = results.multi_hand_landmarks[0]  # Take the first hand
            
            # First pass to get normalization values
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
            
            # Second pass to get normalized coordinates
            min_x = min(x_)
            min_y = min(y_)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)

            if len(data_aux) == 42:  # 21 landmarks * 2 coordinates
                data.append(data_aux)
                labels.append(word_dir)

# Convert to numpy arrays for consistent shape
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

f = open('words_data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"Dataset created with {len(data)} samples")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique words: {np.unique(labels)}") 