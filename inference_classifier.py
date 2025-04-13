import cv2
import pickle
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
mp_draw = mp.solutions.drawing_utils

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            data_aux = []
            x_ = []
            y_ = []
            
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)
                
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.extend([x, y])
                
            # Normalize coordinates
            x1 = min(x_)
            x2 = max(x_)
            y1 = min(y_)
            y2 = max(y_)
            
            for i in range(0, len(data_aux), 2):
                data_aux[i] = (data_aux[i] - x1) / (x2 - x1)
                data_aux[i + 1] = (data_aux[i + 1] - y1) / (y2 - y1)
            
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            
            # Get prediction probabilities
            proba = model.predict_proba([np.asarray(data_aux)])
            confidence = proba[0][model.classes_.tolist().index(predicted_character)]
            
            # Draw prediction and confidence
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            
            cv2.putText(frame, predicted_character, (x1, y1 - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{confidence:.2%}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
