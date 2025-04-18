from flask import Flask, render_template, Response, request, send_from_directory, jsonify, redirect, url_for
import cv2
import os
import json
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from dotenv import load_dotenv, set_key
from ai import process_img_llm, process_img_llm_chemistry, process_text_chemistry_problem, process_text_physics_problem, process_img_llm_physics, process_text_math_problem

app = Flask(__name__)
camera = cv2.VideoCapture(0)
capture_folder = 'captured_images'

load_dotenv()

azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model = os.getenv('AZURE_OPENAI_MODEL')
index = os.getenv('AZURE_OPENAI_INDEX')

vector_store_id = os.getenv('VECTOR_STORE_ID') 

if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{azure_endpoint}/openai/deployments/{model}",
)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action_type = request.args.get('type')
        
        if action_type == 'text':
            # Handle text problem submission
            math_problem = request.form.get('math_problem')
            if math_problem:
                result = process_text_math_problem(math_problem)
                print(f"Result from process_text_math_problem: {result}")  # Debugging output
                return render_template('index.html', img_name=None, result=result)
        
        elif action_type == 'capture':
            # Handle capturing an image from the camera
            success, frame = camera.read()
            if success:
                img_name = os.path.join(capture_folder, "captured_math_image.jpg")
                cv2.imwrite(img_name, frame)
                result = process_img_llm(img_name)
                return render_template('index.html', img_name="captured_math_image.jpg", result=result['formatted_summary'])
            else:
                return "Failed to capture image"
    
    # Handle GET request
    return render_template('index.html', img_name=None, result=None)

def gen_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        img_name = os.path.join(capture_folder, "captured_image.jpg")
        cv2.imwrite(img_name, frame)
        result = process_img_llm(img_name)
        return render_template('captured.html', img_name="captured_image.jpg", result=result['formatted_summary'])
    else:
        return "Failed to capture image"
    
@app.route('/process_image', methods=['POST'])
def process_image():
    img_name = request.json.get('img_name')
    if not img_name:
        return jsonify({'error': 'No image name provided'}), 400

    try:
        result = process_img_llm(img_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/uploads/<filename>')
def uploads(filename):
    return send_from_directory(capture_folder, filename)

@app.route('/chemistry', methods=['GET', 'POST'])
def chemistry():
    if request.method == 'POST':
        action_type = request.args.get('type')
        
        if action_type == 'text':
            # Handle text problem submission
            chemistry_problem = request.form.get('chemistry_problem')
            if chemistry_problem:
                result = process_text_chemistry_problem(chemistry_problem)
                return render_template('chemistry.html', img_name=None, result=result)
        
        elif action_type == 'capture':
            # Handle capturing an image from the camera
            success, frame = camera.read()
            if success:
                img_name = os.path.join(capture_folder, "captured_chemistry_image.jpg")
                cv2.imwrite(img_name, frame)
                result = process_img_llm_chemistry(img_name)
                return render_template('chemistry.html', img_name="captured_chemistry_image.jpg", result=result['formatted_summary'])
            else:
                return "Failed to capture image"
    
    # Handle GET request
    return render_template('chemistry.html', img_name=None, result=None)

@app.route('/physics', methods=['GET', 'POST'])
def physics():
    if request.method == 'POST':
        action_type = request.args.get('type')
        
        if action_type == 'text':
            # Handle text problem submission
            physics_problem = request.form.get('physics_problem')
            if physics_problem:
                result = process_text_physics_problem(physics_problem)
                return render_template('physics.html', img_name=None, result=result)
        
        elif action_type == 'capture':
            # Handle capturing an image from the camera
            success, frame = camera.read()
            if success:
                img_name = os.path.join(capture_folder, "captured_physics_image.jpg")
                cv2.imwrite(img_name, frame)
                result = process_img_llm_physics(img_name)
                return render_template('physics.html', img_name="captured_physics_image.jpg", result=result['formatted_summary'])
            else:
                return "Failed to capture image"
    
    # Handle GET request
    return render_template('physics.html', img_name=None, result=None)



if __name__ == '__main__':
    app.run(debug=True)