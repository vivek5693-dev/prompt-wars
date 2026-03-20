# 🧑‍⚕️ MediBridge - AI Medical Information Extractor

## Overview
MediBridge is a full-stack, smart dynamic assistant designed to instantly convert unstructured, messy medical inputs—like handwritten prescriptions or clinical notes—into structured, actionable, and secure emergency medical data.

## 1. Chosen Vertical
**Healthcare & Emergency Medical Services**
This solution was built around the persona of an emergency room intake nurse or a primary care physician who needs to rapidly digitize complex patient data accurately.

## 2. Approach and Logic
The logic of the application centers on using a minimal, highly-efficient Python Flask backend as a secure integration layer between the user and Google's cutting-edge `gemini-2.0-flash` multimodal model. 
1. The frontend strictly uses standard, accessible HTML5 and Tailwind CSS, enforcing high contrast and ARIA labels.
2. When a user uploads an image or pastes raw medical notes, the application constructs a rigorous structural prompt context.
3. The logic leverages the Google GenAI SDK's OCR and natural language understanding capabilities to extract specific keys: `patient_name`, `blood_type`, `allergies`, `medications`, `conditions`, and `emergency_actions`. 
4. The system validates this JSON and renders a dynamic Emergency Medical Card UI.

## 3. How the Solution Works
1. **User Input:** A user drags and drops a medical document image or pastes text into the accessible UI.
2. **Secure Validation:** The Flask Backend intercepts the form data, enforcing strict 10MB file size limits and mimetype verifications (`RequestEntityTooLarge` handling) to ensure security and efficiency.
3. **Google Services Integration:** The data is seamlessly passed securely to the `gemini-2.0-flash` endpoint.
4. **Data Extraction & Formatting:** The model returns a structured JSON string. The server parses and strictly validates this JSON before responding to the client.
5. **Client Export:** The interface displays the data. Users can copy the formatted JSON or use the integrated `html2pdf.js` library to download a print-ready emergency card. A comprehensive Pytest suite (`tests/test_app.py`) mathematically validates the health and analyze endpoints continuously.

## 4. Assumptions Made
- The environment configuration provides a valid `GEMINI_API_KEY` (Pay-As-You-Go or Free Tier enabled) and is configured to a region supporting `gemini-2.0-flash`.
- The user is uploading documents containing legible English text or internationally recognizable medical shorthand.
- The extracted data is for assistive purposes and assumes a professional will physically verify the output (as clearly stated in the UI security disclaimer). 
- The project runs on standard port `8080`, rendering it pre-configured and optimized for Google Cloud Run deployments.
