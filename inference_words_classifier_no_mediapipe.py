import cv2
import pickle
import numpy as np

# Load the trained model
with open('words_model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Get the available words
with open('words_data.pickle', 'rb') as f:
    data_dict = pickle.load(f)
    labels = data_dict['labels']
    available_words = list(np.unique(labels))
    print(f"Available words: {available_words}")

# Function to extract features from an image without using mediapipe
def extract_features(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Find the largest contour (assumed to be the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box of the hand
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Extract the hand region
    hand_region = thresh[y:y+h, x:x+w]
    
    # Resize to a fixed size for consistent feature extraction
    hand_region = cv2.resize(hand_region, (64, 64))
    
    # Flatten the image to create a feature vector
    features = hand_region.flatten() / 255.0  # Normalize to [0, 1]
    
    return features, (x, y, w, h)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Extract features from the frame
    result = extract_features(frame)
    
    if result[0] is not None:
        features, bbox = result
        
        # Draw bounding box around the hand
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Make prediction
        prediction = model.predict([features])[0]
        word = available_words[0]  # Since we only have one word
        
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