import pytest
import io
import json
from unittest.mock import patch, MagicMock

# Import the application factory
from app import create_app

@pytest.fixture
def app():
    _app = create_app()
    _app.config['TESTING'] = True
    return _app

@pytest.fixture
def test_client(app):
    return app.test_client()

def test_health_check(test_client):
    response = test_client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "services" in data

def test_analyze_no_input(test_client):
    response = test_client.post('/analyze', data={})
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_analyze_invalid_file_type(test_client):
    data = {'file': (io.BytesIO(b"dummy data"), 'test.txt')}
    response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert "Invalid file type" in response.get_json()["error"]

@patch('app.GeminiMedicalExtractor.analyze_contents')
def test_analyze_success_with_text(mock_analyze, test_client):
    # Setup mock response output correctly formatted
    mock_analyze.return_value = {
        "patient_name": "Test Name", 
        "blood_type": "O-", 
        "allergies": ["Peanuts"], 
        "medications": [], 
        "conditions": [], 
        "emergency_actions": [], 
        "summary": "Test Summary"
    }
    
    data = {
        'text': 'Patient Test Name is allergic to Peanuts. Blood type O-.'
    }
    
    response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    res_data = response.get_json()
    assert res_data['patient_name'] == "Test Name"
    assert "Peanuts" in res_data['allergies']

def test_analyze_file_too_large(app, test_client):
    # Temporarily set max length to tiny size
    app.config['MAX_CONTENT_LENGTH'] = 10  # 10 bytes
    large_data = b"0" * 20
    data = {'file': (io.BytesIO(large_data), 'test.jpg')}
    
    response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 413
    assert "error" in response.get_json()
