import os
import cv2

# Create data directory if it doesn't exist
DATA_DIR = './words_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Word to collect data for
WORD = "HELLO"  # You can change this to any word you want to collect

# Create directory for the word if it doesn't exist
word_dir = os.path.join(DATA_DIR, WORD)
if not os.path.exists(word_dir):
    os.makedirs(word_dir)

# Number of samples to collect
dataset_size = 50

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print(f'Collecting data for word: {WORD}')
print('Press "q" to start collecting data')

# Wait for user to press 'q' to start
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    cv2.putText(frame, 'Ready? Press "Q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.putText(frame, f'Word: {WORD}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print('Starting data collection...')

# Collect data
counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    cv2.putText(frame, f'Word: {WORD}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.putText(frame, f'Capturing: {counter}/{dataset_size}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                cv2.LINE_AA)
    cv2.imshow('frame', frame)
    
    # Save the frame
    cv2.imwrite(os.path.join(word_dir, f'{counter}.jpg'), frame)
    
    # Wait for a short time
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    counter += 1

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
print(f"Collected {counter} samples for word: {WORD}")
print("Now run 'python create_words_dataset_no_mediapipe.py' to process the images")
print("Then run 'python train_words_classifier.py' to train the model")
print("Finally, run 'python inference_words_classifier_no_mediapipe.py' to test the model") 