import cv2
import pickle
import numpy as np
import mediapipe as mp
import time

# Load the model and scaler
print("Loading model and scaler...")
model = pickle.load(open('./model.p', 'rb'))  # Load model directly
scaler = pickle.load(open('./scaler.p', 'rb'))

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize text buffer
text_buffer = ""
last_prediction_time = time.time()
prediction_cooldown = 0.5  # seconds
last_predicted_char = None

print("\nStarting sign language typer...")
print("Controls:")
print("- Make hand signs for letters (A-Z)")
print("- Press SPACE to add a space")
print("- Press BACKSPACE to delete last character")
print("- Press ENTER to save phrase")
print("- Press ESC to quit")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks = results.multi_hand_landmarks[0]
        
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
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get bounding box coordinates
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make prediction
            data_aux = np.asarray(data_aux).reshape(1, -1)
            # Apply the same scaling as during training
            data_aux = scaler.transform(data_aux)
            prediction = model.predict(data_aux)
            probabilities = model.predict_proba(data_aux)
            predicted_class = int(prediction[0])
            confidence = probabilities[0][predicted_class] * 100

            # Only add character if confidence is high enough and cooldown has passed
            current_time = time.time()
            if (confidence > 60 and 
                current_time - last_prediction_time > prediction_cooldown and 
                predicted_class != last_predicted_char):
                predicted_char = chr(65 + predicted_class)  # Convert to letter (A=0, B=1, etc.)
                text_buffer += predicted_char
                last_prediction_time = current_time
                last_predicted_char = predicted_class

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, f"{chr(65 + predicted_class)} ({confidence:.1f}%)", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    # Display current text buffer
    cv2.putText(frame, f"Text: {text_buffer}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Typer', frame)
    key = cv2.waitKey(1)
    
    # Handle keyboard input
    if key == ord(' '):  # Space
        text_buffer += " "
    elif key == ord('\b'):  # Backspace
        text_buffer = text_buffer[:-1]
    elif key == ord('\r'):  # Enter
        if text_buffer.strip():  # Only save if there's text
            with open('saved_phrases.txt', 'a') as f:
                f.write(text_buffer + '\n')
            print(f"\nSaved phrase: {text_buffer}")
            text_buffer = ""
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows() 