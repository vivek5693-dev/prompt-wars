import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

# Config
# 10 MB limit
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic'}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Gemini Client
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None
    print("WARNING: GEMINI_API_KEY is not set.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if not client:
        return jsonify({"error": "Gemini API key not configured"}), 500

    text_input = request.form.get('text', '')
    
    file = request.files.get('file')
    
    if not file and not text_input.strip():
        return jsonify({"error": "Please provide either an image or medical text"}), 400

    contents = []
    
    # Check if a file is provided
    if file and file.filename != '':
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Must be a PNG, JPG, JPEG, WEBP or HEIC image."}), 400
        
        # Read file bytes for Gemini
        file_bytes = file.read()
        mime_type = file.mimetype or 'image/jpeg'
        
        try:
            image_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            contents.append(image_part)
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
        
    if text_input.strip():
        contents.append(text_input.strip())

    system_instruction = (
        "You are a medical information extraction assistant. Analyze the following medical document/text and extract structured information. "
        "Return ONLY valid JSON with these fields: "
        "patient_name (string), blood_type (string), allergies (array of strings), "
        "medications (array of objects with 'name' and 'dosage' keys), "
        "conditions (array of strings), "
        "emergency_actions (array of strings representing critical warnings or actions), "
        "summary (string). "
        "If a field is not found in the provided input, use null for that field. Be thorough and accurate."
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        
        # Parse output to ensure it's valid JSON before sending
        try:
            result_json = json.loads(response.text)
            return jsonify(result_json), 200
        except json.JSONDecodeError:
            return jsonify({"error": "Failed to parse JSON from AI model", "raw": response.text}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB."}), 413

@app.errorhandler(Exception)
def handle_general_exception(e):
    if isinstance(e, RequestEntityTooLarge):
        return handle_file_too_large(e)
    # Log the exception stack trace to console
    import traceback
    traceback.print_exc()
    return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
