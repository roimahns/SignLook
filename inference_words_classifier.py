import cv2
import pickle
import numpy as np
import mediapipe as mp

# Load the trained model
with open('words_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Dictionary to map class indices to words
with open('words_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    labels = data_dict['labels']
    unique_labels = np.unique(labels)
    label_to_word = {i: str(word) for i, word in enumerate(unique_labels)}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit")
print(f"Available words: {unique_labels}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(rgb_frame)
    
    # Initialize list to store hand landmarks
    hand_landmarks_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and normalize landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            # Normalize landmarks
            landmarks = np.array(landmarks)
            min_val = landmarks.min()
            max_val = landmarks.max()
            if max_val - min_val > 0:
                landmarks = (landmarks - min_val) / (max_val - min_val)
            
            hand_landmarks_list.append(landmarks)
    
    # Make prediction if hands are detected
    if hand_landmarks_list:
        # Use the first hand detected
        prediction = model.predict([hand_landmarks_list[0]])[0]
        word = label_to_word[prediction]
        
        # Display the predicted word
        cv2.putText(frame, f'Word: {word}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('ASL Word Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close() 