import pytest
import io
import json
from unittest.mock import patch, MagicMock

# Import the Flask application instance
from app import app

@pytest.fixture
def test_client():
    app.config['TESTING'] = True
    # Ensure errors are propagated to response but we can also handle standard routing
    with app.test_client() as client:
        yield client

def test_health_check(test_client):
    response = test_client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"

def test_analyze_no_input(test_client):
    response = test_client.post('/analyze', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "either an image or medical text" in data["error"]

def test_analyze_invalid_file_type(test_client):
    data = {
        'file': (io.BytesIO(b"fake txt content"), 'test.txt')
    }
    response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "File type not allowed" in data["error"]

def test_analyze_success_with_text(test_client):
    with patch('app.client') as mock_client:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "patient_name": "Test Name", 
            "blood_type": "O-", 
            "allergies": ["Peanuts"], 
            "medications": [], 
            "conditions": [], 
            "emergency_actions": [], 
            "summary": "Test Summary"
        })
        
        # Configure the mock chain
        mock_client.models.generate_content.return_value = mock_response
        
        # Because we're overriding global app.client, we need to temporarily mock it 
        # inside the app module level
        import app as my_app
        old_client = my_app.client
        my_app.client = mock_client
        
        data = {
            'text': 'Patient Test Name is allergic to Peanuts. Blood type O-.'
        }
        
        response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
        
        # Restore client
        my_app.client = old_client
        
        assert response.status_code == 200
        res_data = response.get_json()
        assert res_data['patient_name'] == "Test Name"
        assert "Peanuts" in res_data['allergies']

def test_analyze_file_too_large(test_client):
    # Test setting a very small limit just for the test
    app.config['MAX_CONTENT_LENGTH'] = 100 # 100 bytes
    large_content = b"0" * 200 # Over the limit
    data = {
        'file': (io.BytesIO(large_content), 'large.png')
    }
    
    response = test_client.post('/analyze', data=data, content_type='multipart/form-data')
    
    # Restore limit just in case
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
    
    assert response.status_code == 413
    assert "File too large" in response.get_json()['error']
