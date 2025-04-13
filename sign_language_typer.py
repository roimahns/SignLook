import cv2
import pickle
import numpy as np
import mediapipe as mp
import time
import os

# Load the model and scaler
print("Loading model and scaler...")
model = pickle.load(open('./model.p', 'rb'))  # Load model directly
scaler = pickle.load(open('./scaler.p', 'rb'))

# Get the number of classes from the model
n_classes = len(model.classes_)
print(f"Model has {n_classes} classes")

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
status_message = "Show your hand to the camera"
status_timer = time.time()
status_duration = 2.0  # seconds
current_letter = None
current_confidence = 0
is_thinking = False
thinking_timer = time.time()
last_key_time = time.time()
key_cooldown = 0.3  # seconds to prevent multiple key presses
auto_save = False  # Disable auto-save feature

# Create saved_phrases directory if it doesn't exist
if not os.path.exists('saved_phrases'):
    os.makedirs('saved_phrases')

# UI Colors
COLOR_BG = (40, 40, 40)  # Dark gray background
COLOR_TEXT = (255, 255, 255)  # White text
COLOR_HIGHLIGHT = (0, 255, 0)  # Green highlight
COLOR_WARNING = (0, 0, 255)  # Red warning
COLOR_INFO = (255, 165, 0)  # Orange info
COLOR_STATUS = (255, 255, 0)  # Yellow status
COLOR_THINKING = (0, 191, 255)  # Deep sky blue for thinking

print("\nStarting sign language typer...")
print("Controls:")
print("- Make hand signs for letters (A-Z)")
print("- Press ENTER to add the detected letter")
print("- Press SPACE to add a space")
print("- Press BACKSPACE to delete last character")
print("- Press S to save phrase")
print("- Press ESC to quit")
print("- Phrases are saved to saved_phrases.txt when you press S")

# Function to save the current phrase
def save_phrase():
    if text_buffer.strip():  # Only save if there's text
        with open('saved_phrases.txt', 'a') as f:
            f.write(text_buffer + '\n')
        print(f"\nSaved phrase: {text_buffer}")
        return f"Saved phrase: {text_buffer}"
    return "No text to save"

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
    
    # Create a dark overlay for better text visibility
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 100), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Default status message
    current_status = "Show your hand to the camera"
    is_thinking = False
    
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
            # Update status message
            current_status = "Processing hand gesture..."
            is_thinking = True
            thinking_timer = time.time()
            
            # Draw the hand landmarks with custom colors
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
            
            # Ensure predicted_class is within bounds
            if predicted_class < n_classes:
                confidence = probabilities[0][predicted_class] * 100
                current_confidence = confidence
                current_letter = chr(65 + predicted_class)  # Convert to letter (A=0, B=1, etc.)

                # Update status based on confidence - removed the "STAY STILL" message
                current_status = f"Press ENTER to add: {current_letter} ({confidence:.1f}%)"

                # Draw rectangle and label with better styling
                # Draw a thicker, more visible box around the hand
                box_color = COLOR_THINKING if is_thinking else COLOR_HIGHLIGHT
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Add a semi-transparent background for the label
                label_bg = frame.copy()
                cv2.rectangle(label_bg, (x1, y1 - 30), (x1 + 120, y1), COLOR_BG, -1)
                cv2.addWeighted(label_bg, 0.7, frame, 0.3, 0, frame)
                
                # Draw the prediction label
                cv2.putText(frame, f"{current_letter} ({confidence:.1f}%)", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2,
                            cv2.LINE_AA)
                
                # Draw a small indicator dot based on confidence
                dot_color = COLOR_HIGHLIGHT if confidence > 60 else COLOR_WARNING
                cv2.circle(frame, (x2 + 10, y1 + 10), 5, dot_color, -1)
                
                # Draw thinking animation if processing
                if is_thinking:
                    # Draw a pulsing circle
                    pulse_size = int(5 + 3 * np.sin(time.time() * 5))
                    cv2.circle(frame, (x2 + 30, y1 + 10), pulse_size, COLOR_THINKING, -1)
    
    # Display status message
    if time.time() - status_timer < status_duration:
        # Show temporary status message
        cv2.putText(frame, status_message, (W//2 - 150, H - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_STATUS, 2, cv2.LINE_AA)
    else:
        # Show current status
        cv2.putText(frame, current_status, (W//2 - 200, H - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_STATUS, 2, cv2.LINE_AA)

    # Display current text buffer with better styling
    cv2.putText(frame, "Current Text:", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    
    # Fix text display by ensuring it's visible and properly formatted
    if text_buffer:
        # Draw a background for the text to make it more visible
        text_bg = frame.copy()
        text_width = len(text_buffer) * 15  # Approximate width based on character count
        cv2.rectangle(text_bg, (10, 40), (10 + text_width, 70), COLOR_BG, -1)
        cv2.addWeighted(text_bg, 0.7, frame, 0.3, 0, frame)
        
        # Draw the text with a larger font and thicker lines
        cv2.putText(frame, text_buffer, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
    else:
        # Show placeholder text when buffer is empty
        cv2.putText(frame, "<empty>", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFO, 2, cv2.LINE_AA)

    # Display FPS with better styling
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps_color = COLOR_HIGHLIGHT if fps >= 25 else COLOR_WARNING
    cv2.putText(frame, f"FPS: {fps}", (W - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

    # Display controls hint
    cv2.putText(frame, "ENTER=Add Letter  SPACE=Space  BACKSPACE=Delete  S=Save  ESC=Quit", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO, 1, cv2.LINE_AA)

    cv2.imshow('Sign Language Typer', frame)
    
    # Improved key detection with cooldown
    current_time = time.time()
    if current_time - last_key_time > key_cooldown:
        key = cv2.waitKey(1) & 0xFF  # Use bitwise AND to get the correct key code
        
        # Handle keyboard input
        if key == 13 or key == ord('\r'):  # Enter key (check both codes)
            if current_letter:  # Removed confidence threshold check
                text_buffer += current_letter
                status_message = f"Added letter: {current_letter}"
                status_timer = time.time()
                last_key_time = current_time
                print(f"Added letter: {current_letter}")  # Debug print
        elif key == ord(' '):  # Space
            text_buffer += " "
            status_message = "Added space"
            status_timer = time.time()
            last_key_time = current_time
        elif key == ord('\b'):  # Backspace
            text_buffer = text_buffer[:-1]
            status_message = "Deleted last character"
            status_timer = time.time()
            last_key_time = current_time
        elif key == ord('s'):  # Save
            status_message = save_phrase()
            status_timer = time.time()
            text_buffer = ""
            last_key_time = current_time
        elif key == 27:  # ESC
            # Save any remaining text before quitting
            if text_buffer.strip():
                save_phrase()
            break
    
    # Check for Enter key using a different method (for redundancy)
    if current_time - last_key_time > key_cooldown:
        try:
            # Try to detect Enter key using a different method
            if cv2.getWindowProperty('Sign Language Typer', cv2.WND_PROP_VISIBLE) < 1:
                break
                
            # Check if Enter key is pressed using a different method
            key_state = cv2.waitKey(1) & 0xFF
            if key_state == 13 or key_state == ord('\r'):
                if current_letter:  # Removed confidence threshold check
                    text_buffer += current_letter
                    status_message = f"Added letter: {current_letter}"
                    status_timer = time.time()
                    last_key_time = current_time
                    print(f"Added letter: {current_letter} (alternative method)")  # Debug print
        except:
            pass

cap.release()
cv2.destroyAllWindows() 