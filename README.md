# Sign Language Recognition Web App

A web-based application for real-time sign language recognition using your computer's camera.

## Features

- Real-time hand gesture recognition
- Manual letter confirmation with Enter key
- Prediction stabilization for more reliable detection
- Save and view your typed phrases
- Modern, user-friendly interface

## Setup

1. Install Python 3.8 or higher
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have the following files in your project directory:
   - `model.p` (trained model)
   - `scaler.p` (fitted StandardScaler)
   - `app.py` (Flask application)
   - `templates/index.html` (HTML template)
   - `static/app.js` (JavaScript code)

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Allow camera access when prompted
2. Show your hand to the camera
3. Hold your hand steady until the prediction stabilizes
4. Press Enter to add the detected letter to your phrase
5. Use the controls to:
   - Add letters (Enter)
   - Delete characters (Backspace)
   - Add spaces (Space)
   - Save phrases (Save)

## Controls

- **Enter**: Add the current letter to your phrase
- **Backspace**: Delete the last character
- **Space**: Add a space to your phrase
- **Save**: Save the current phrase

## Notes

- Make sure you have good lighting for better hand detection
- Keep your hand within the camera frame
- Hold your hand steady for more accurate predictions
- The system requires a prediction to be stable (appear multiple times) before accepting it
