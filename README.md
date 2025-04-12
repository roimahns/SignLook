# SignLook - ASL Sign Language Detector

A Python-based application that uses computer vision to detect and recognize American Sign Language (ASL) signs in real-time. This project supports both alphabet signs and common words/phrases.

## Features

- Real-time hand detection and tracking
- ASL alphabet recognition
- Common words/phrases recognition
- Easy-to-use interface with webcam support
- High accuracy using machine learning models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/roimahns/SignLook.git
cd SignLook
```

2. Create a virtual environment (Python 3.11 recommended):
```bash
python -m venv venv311
.\venv311\Scripts\activate  # Windows
source venv311/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install opencv-python mediapipe numpy scikit-learn matplotlib seaborn
```

## Usage

### Alphabet Recognition
```bash
python collect_imgs.py  # Collect training data
python create_dataset.py  # Process the collected images
python train_classifier.py  # Train the model
python inference_classifier.py  # Run real-time recognition
```

### Word Recognition
```bash
python collect_words.py  # Collect word training data
python create_words_dataset.py  # Process the word images
python train_words_classifier.py  # Train the word model
python inference_words_classifier.py  # Run real-time word recognition
```

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
