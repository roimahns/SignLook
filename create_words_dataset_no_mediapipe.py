import os
import pickle
import cv2
import numpy as np

DATA_DIR = './words_data'

data = []
labels = []

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
        return None
    
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
    
    return features

for word_dir in os.listdir(DATA_DIR):
    if not os.path.isdir(os.path.join(DATA_DIR, word_dir)):
        continue
        
    print(f"Processing word: {word_dir}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, word_dir)):
        img = cv2.imread(os.path.join(DATA_DIR, word_dir, img_path))
        if img is None:
            continue
        
        # Extract features from the image
        features = extract_features(img)
        
        if features is not None:
            data.append(features)
            labels.append(word_dir)

# Convert to numpy arrays for consistent shape
data = np.array(data, dtype=np.float32)
labels = np.array(labels)

# Save the dataset
f = open('words_data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"Dataset created with {len(data)} samples")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique words: {np.unique(labels)}") 