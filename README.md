# Math, Chemistry, and Physics Assistant

This is a Flask web application designed to solve **Math**, **Chemistry**, and **Physics** problems using **Azure OpenAI**. The app provides a dynamic and interactive interface for users to either type their problems or capture them using a webcam, and it returns detailed solutions.

---

## Features

1. **Math Problem Solver**:
   - Users can type math problems (e.g., equations, algebraic expressions) into a text form.
   - Alternatively, users can capture a photo of a math problem using their device's camera.
   - The app processes the input and provides a detailed solution.

2. **Chemistry Problem Solver**:
   - Users can type chemistry-related questions (e.g., molar mass calculations, balancing equations) or upload a photo of a chemistry problem.
   - The app identifies the problem and provides a step-by-step solution.
   - Supports multiple-choice questions by identifying the correct answer and explaining it.

3. **Physics Problem Solver**:
   - Users can type physics problems (e.g., force, acceleration, energy calculations) or capture a photo of a physics problem.
   - The app provides detailed solutions, including formulas and calculations.
   - Handles multiple-choice questions with correct answers and explanations.

4. **Live Camera Feed**:
   - A live video feed is displayed on the page, allowing users to preview the problem before capturing it.

5. **Dynamic and Interactive Interface**:
   - Separate pages for math, chemistry, and physics, each tailored to the specific subject.
   - Results are displayed dynamically on the same page after submission.

---

## How It Works

1. **Text Input**:
   - Users type their problem into the provided text form.
   - The app sends the problem to the Azure OpenAI API for processing and displays the solution.

2. **Image Capture**:
   - Users capture a photo of the problem using their device's camera.
   - The app processes the image, extracts the problem, and sends it to the Azure OpenAI API for analysis.

3. **Azure OpenAI Integration**:
   - The app uses Azure OpenAI to analyze and solve problems.
   - The API generates detailed solutions in plain text or JSON format.

---

## Technologies Used

- **Flask**: Backend framework for handling routes and rendering templates.
- **Azure OpenAI**: AI-powered problem-solving engine.
- **OpenCV**: For capturing and processing images from the camera.
- **HTML/CSS**: For building the user interface.
- **JavaScript**: For dynamic form actions and live camera feed integration.
- **Jinja2**: For rendering dynamic content in templates.

---

Folder Structure

MathApp/
├── app.py                 # Main Flask application
├── ai.py                  # AI processing functions for math, chemistry, and physics
├── templates/
│   ├── index.html         # Math page template
│   ├── chemistry.html     # Chemistry page template
│   ├── physics.html       # Physics page template
├── static/
│   ├── style.css          # CSS for styling the app
├── captured_images/       # Folder for storing captured images
├── .env                   # Environment variables for Azure OpenAI API

## Requirements

- Python 3.8+
- Flask
- OpenCV
- Azure OpenAI API access
- dotenv

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/MathApp.git
   cd MathApp
2. Install dependencies:
    pip install -r requirements.txt

3. Set up the .env file with your Azure OpenAI API credentials:
    
    AZURE_OPENAI_API_ENDPOINT=<your_endpoint>

    AZURE_OPENAI_API_KEY=<your_api_key>

    AZURE_OPENAI_API_VERSION=<api_version>
    
    AZURE_OPENAI_MODEL=<model_name>

4. Run the application:
    python app.py

5. Open your browser and navigate to:
    http://127.0.0.1:5000/

Future Enhancements
* Add support for additional subjects like biology or engineering.
* Improve image processing for handwritten problems.
* Add user authentication for personalized problem-solving history.