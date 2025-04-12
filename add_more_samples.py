import os
import sys

import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Default to collecting all letters if no arguments provided
if len(sys.argv) > 1:
    # Parse the letters provided as arguments
    letters_to_collect = []
    for arg in sys.argv[1:]:
        if len(arg) == 1 and arg.isalpha():
            letters_to_collect.append(ord(arg.upper()) - 65)  # Convert A-Z to 0-25
        elif arg.isdigit() and 0 <= int(arg) <= 25:
            letters_to_collect.append(int(arg))
    
    if not letters_to_collect:
        print("No valid letters provided. Please provide letters A-Z or numbers 0-25.")
        sys.exit(1)
    
    print(f"Adding more samples for letters: {[chr(65 + i) for i in letters_to_collect]}")
else:
    # If no arguments, ask the user which letters to collect
    print("Which letters would you like to add more samples for? (Enter letters separated by spaces, e.g., 'A B C' or '0 1 2')")
    user_input = input("> ")
    letters_to_collect = []
    
    for arg in user_input.split():
        if len(arg) == 1 and arg.isalpha():
            letters_to_collect.append(ord(arg.upper()) - 65)  # Convert A-Z to 0-25
        elif arg.isdigit() and 0 <= int(arg) <= 25:
            letters_to_collect.append(int(arg))
    
    if not letters_to_collect:
        print("No valid letters provided. Please provide letters A-Z or numbers 0-25.")
        sys.exit(1)
    
    print(f"Adding more samples for letters: {[chr(65 + i) for i in letters_to_collect]}")

# Ask how many samples to add
print("How many samples would you like to add for each letter? (default: 50)")
samples_input = input("> ")
try:
    dataset_size = int(samples_input) if samples_input.strip() else 50
except ValueError:
    dataset_size = 50
    print(f"Invalid input. Using default value of {dataset_size} samples.")

cap = cv2.VideoCapture(0)
for j in letters_to_collect:
    # Create directory if it doesn't exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Convert class number to letter (0=A, 1=B, ..., 25=Z)
    letter = chr(65 + j)  # ASCII: 65 is 'A'
    
    # Count existing samples
    existing_samples = len([f for f in os.listdir(os.path.join(DATA_DIR, str(j))) if f.endswith('.jpg')])
    print(f'Letter {letter} has {existing_samples} existing samples. Adding {dataset_size} more.')

    done = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Letter: {}'.format(letter), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.putText(frame, 'Letter: {}'.format(letter), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Capturing: {}/{}'.format(counter, dataset_size), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Save with a new index to avoid overwriting
        new_index = existing_samples + counter
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(new_index)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

print("Additional samples collected successfully!")
print("Now run 'python create_dataset.py' to process the images")
print("Then run 'python train_classifier.py' to train the model")
print("Finally, run 'python inference_classifier.py' to test the model") 