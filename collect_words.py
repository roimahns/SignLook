import os
import sys

import cv2


DATA_DIR = './words_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Common ASL words to collect
COMMON_WORDS = [
    "HELLO", "THANK YOU", "PLEASE", "SORRY", "YES", "NO", 
    "HELP", "WATER", "FOOD", "BATHROOM", "GOOD", "BAD",
    "MORNING", "NIGHT", "FRIEND", "FAMILY", "LOVE", "HAPPY",
    "SAD", "HUNGRY", "THIRSTY", "TIRED", "SICK", "HURT"
]

# Default to collecting all words if no arguments provided
if len(sys.argv) > 1:
    # Parse the words provided as arguments
    words_to_collect = []
    for arg in sys.argv[1:]:
        word = arg.upper()
        if word in COMMON_WORDS:
            words_to_collect.append(word)
    
    if not words_to_collect:
        print("No valid words provided. Please provide words from the list.")
        print("Available words:", ", ".join(COMMON_WORDS))
        sys.exit(1)
    
    print(f"Collecting data for words: {words_to_collect}")
else:
    # If no arguments, ask the user which words to collect
    print("Which words would you like to collect data for? (Enter words separated by spaces)")
    print("Available words:", ", ".join(COMMON_WORDS))
    user_input = input("> ")
    words_to_collect = []
    
    for word in user_input.split():
        word = word.upper()
        if word in COMMON_WORDS:
            words_to_collect.append(word)
    
    if not words_to_collect:
        print("No valid words provided. Please provide words from the list.")
        sys.exit(1)
    
    print(f"Collecting data for words: {words_to_collect}")

# Ask how many samples to collect per word
print("How many samples would you like to collect for each word? (default: 50)")
samples_input = input("> ")
try:
    dataset_size = int(samples_input) if samples_input.strip() else 50
except ValueError:
    dataset_size = 50
    print(f"Invalid input. Using default value of {dataset_size} samples.")

cap = cv2.VideoCapture(0)
for word in words_to_collect:
    # Create directory if it doesn't exist
    word_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)

    print(f'Collecting data for word: {word}')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Word: {word}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, f'Word: {word}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Capturing: {counter}/{dataset_size}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(word_dir, f'{counter}.jpg'), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
print("Now run 'python create_words_dataset.py' to process the images")
print("Then run 'python train_words_classifier.py' to train the model")
print("Finally, run 'python inference_words_classifier.py' to test the model") 