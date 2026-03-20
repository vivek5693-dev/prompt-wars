"""
MediBridge - AI Medical Information Extractor
A Gemini-powered application for converting unstructured medical inputs into life-saving actions.
"""
import os
import uuid
import logging
from typing import Tuple, Dict, Any

from flask import Flask, request, jsonify, render_template, Response
from flask_compress import Compress
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from google.genai import types

# Import modular services and models
from config import Config
from services.logging_service import LoggingService
from services.storage_service import StorageService
from services.gemini_service import GeminiService
from models.medical_data import MedicalExtraction

# Initialize structured logging
LoggingService.initialize()

def create_app() -> Flask:
    """Application factory for MediBridge."""
    app = Flask(__name__)
    Compress(app)
    app.config.from_object(Config)

    # Initialize Services
    gemini_service = GeminiService(Config.GEMINI_API_KEY)
    storage_service = StorageService(Config.GCS_BUCKET_NAME)

    def allowed_file(filename: str) -> bool:
        """Validates allowable file extensions."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

    @app.after_request
    def apply_security_headers(response: Response) -> Response:
        """Enforces robust browser security headers globally."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # Adjusted CSP for external scripts used in the UI
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline' "
            "https://cdn.tailwindcss.com "
            "https://cdnjs.cloudflare.com "
            "https://cdn.jsdelivr.net;"
        )
        return response

    @app.route('/')
    def index() -> str:
        return render_template('index.html')

    @app.route('/health', methods=['GET'])
    def health_check() -> Tuple[Response, int]:
        services = {
            "gemini_api": gemini_service.client is not None,
            "cloud_storage": storage_service.client is not None,
            "grounding_enabled": Config.USE_GROUNDING
        }
        return jsonify({"status": "healthy", "services": services}), 200

    @app.route('/analyze', methods=['POST'])
    def analyze() -> Tuple[Response, int]:
        audit_id = str(uuid.uuid4())
        try:
            text_input = request.form.get('text', '')
            file = request.files.get('file')
            
            if not file and not text_input.strip():
                return jsonify({"error": "No input provided. Please upload an image or enter text."}), 400

            contents = []
            
            if file and file.filename != '':
                secure_name = secure_filename(file.filename)
                if not allowed_file(secure_name):
                    return jsonify({"error": "Invalid file format. Please use images (JPG, PNG, WEBP)."}), 400
                
                file_bytes = file.read()
                mime_type = file.mimetype or 'image/jpeg'
                
                # Optional GCS Upload for audit trail/Cloud adoption
                storage_service.upload_file(file_bytes, f"uploads/{audit_id}_{secure_name}", mime_type)
                
                parts = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                contents.append(parts)
                
            if text_input.strip():
                # Sanitize and limit text input length
                contents.append(text_input.strip()[:8000])

            # Analyze using Gemini Service with Grounding
            extraction_result: MedicalExtraction = gemini_service.analyze_medical_contents(
                contents=contents, 
                use_grounding=Config.USE_GROUNDING
            )
            
            LoggingService.info(f"Medical analysis successful for ID: {audit_id}")
            return jsonify(extraction_result.model_dump()), 200
            
        except Exception as e:
            LoggingService.error(f"Analysis Error [ID: {audit_id}]: {str(e)}")
            return jsonify({"error": "Failed to process medical data. Please try again with clearer input."}), 500

    @app.errorhandler(413)
    def handle_file_too_large(e) -> Tuple[Response, int]:
        return jsonify({"error": "File size exceeds the 10MB limit."}), 413

    @app.errorhandler(Exception)
    def handle_general_exception(e) -> Tuple[Response, int]:
        if isinstance(e, RequestEntityTooLarge):
            return handle_file_too_large(e)
        LoggingService.critical(f"Unhandled Exception: {str(e)}")
        return jsonify({"error": "A secure internal error occurred."}), 500

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT, debug=False)
