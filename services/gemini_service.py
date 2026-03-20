import json
import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from models.medical_data import MedicalExtraction

class GeminiService:
    """Service for Google Gemini interactions with Vertex AI features."""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        # Default to gemini-2.0-flash as it's the latest and most capable for multimodal
        self.model_name = "gemini-2.0-flash" 
        self.client = genai.Client(api_key=api_key) if api_key else None

    def analyze_medical_contents(self, contents: List[Any], use_grounding: bool = True) -> MedicalExtraction:
        """
        Analyzes multimodal medical data and returns a structured Pydantic model.
        Uses Google Search Grounding for verification if enabled.
        """
        if not self.client:
            raise ValueError("Gemini API key not configured")

        system_instruction = (
            "You are a highly-secure, life-saving medical information extraction assistant. "
            "Your goal is to bridge the gap between messy, unstructured real-world inputs "
            "and structured, verified emergency medical actions. "
            "Analyze the provided inputs (images/text) and extract medical data accurately. "
            "Focus on 'emergency_actions' that could save a life in an ER setting. "
            "Return ONLY valid JSON that matches the requested schema. "
            "If information is missing, use null or an empty list where appropriate."
        )

        tools = []
        if use_grounding:
            # Enable Google Search Grounding for verification
            tools.append(types.Tool(google_search=types.GoogleSearch()))

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    tools=tools,
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            
            # Parse the response text as JSON
            raw_data = json.loads(response.text)
            
            # Add verification flag if grounding was used
            raw_data["is_verified"] = use_grounding
            
            # Validate with Pydantic
            return MedicalExtraction(**raw_data)
            
        except Exception as e:
            logging.error(f"Gemini Analysis Error: {e}")
            raise
