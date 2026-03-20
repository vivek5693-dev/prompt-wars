import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration class for secure environment variables."""
    PORT = int(os.environ.get('PORT', 8080))
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'heic'}
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
    USE_GROUNDING = os.environ.get("USE_GROUNDING", "true").lower() == "true"
