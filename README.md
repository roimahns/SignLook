# SignLook - Real-Time Sign Language Recognition

A web-based application that converts hand gestures into text in real-time using your computer's camera. Built with Python, OpenCV, and MediaPipe for accurate hand gesture recognition.

## Inspiration

When my best friend recently lost his hearing, I found myself struggling to communicate effectively. This personal challenge inspired me to create SignLook - a tool that bridges the communication gap while I learned sign language myself. The project was built entirely solo during a hackathon, even overcoming challenges like a blackout that had me chasing Wi-Fi signals between libraries and cafés!

## Features

- Real-time hand gesture recognition with 91% accuracy
- Manual letter confirmation with Enter key
- Prediction stabilization for reliable detection
- Save and view your typed phrases
- Modern, user-friendly interface
- Works with standard webcam
- No installation required - runs in your browser

## Technologies Used

- **Backend:**
  - Python 3.8+
  - Flask (Web framework)
  - OpenCV (Computer Vision)
  - MediaPipe (Hand Tracking)
  - Scikit-learn (Machine Learning)
  - NumPy (Data Processing)

- **Frontend:**
  - HTML5
  - JavaScript
  - CSS3

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

## Future Improvements

- Adding support for full words and phrases
- Improving accuracy in different lighting conditions
- Creating a mobile app version
- Adding a learning mode for beginners
- Supporting multiple sign language dialects

## Contributing

Feel free to contribute to this project! Whether you're learning sign language, know someone who's hearing impaired, or just interested in the technology, I hope SignLook can make a difference in your life too.

## License

This project is open source and available under the MIT License.
