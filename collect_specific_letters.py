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
    
    print(f"Collecting data for letters: {[chr(65 + i) for i in letters_to_collect]}")
else:
    # If no arguments, collect all letters
    letters_to_collect = list(range(26))
    print("Collecting data for all letters A-Z")

dataset_size = 100

cap = cv2.VideoCapture(0)
for j in letters_to_collect:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Convert class number to letter (0=A, 1=B, ..., 25=Z)
    letter = chr(65 + j)  # ASCII: 65 is 'A'
    print('Collecting data for letter {}'.format(letter))

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
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
print("Now run 'python create_dataset.py' to process the images")
print("Then run 'python train_classifier.py' to train the model")
print("Finally, run 'python inference_classifier.py' to test the model") 