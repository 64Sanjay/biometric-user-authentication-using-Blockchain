# tests/test_web_app.py
"""
Tests for Flask web application
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create test client for Flask app."""
    from web.app import app
    
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestWebRoutes:
    """Tests for web routes."""
    
    def test_home_page(self, client):
        """Test home page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Biometric' in response.data or b'biometric' in response.data
    
    def test_enroll_page(self, client):
        """Test enrollment page loads."""
        response = client.get('/enroll')
        assert response.status_code == 200
    
    def test_authenticate_page(self, client):
        """Test authentication page loads."""
        response = client.get('/authenticate')
        assert response.status_code == 200
    
    def test_dashboard_page(self, client):
        """Test dashboard page loads."""
        response = client.get('/dashboard')
        assert response.status_code == 200
    
    def test_architecture_page(self, client):
        """Test architecture page loads."""
        response = client.get('/architecture')
        assert response.status_code == 200


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_status_api(self, client):
        """Test status API endpoint."""
        response = client.get('/api/status')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'success' in data
    
    def test_enroll_api(self, client):
        """Test enroll API endpoint."""
        response = client.post('/api/enroll', 
            json={'user_id': 'test_api_user'},
            content_type='application/json'
        )
        assert response.status_code == 200
    
    def test_authenticate_api(self, client):
        """Test authenticate API endpoint."""
        response = client.post('/api/authenticate',
            json={'user_id': 'test_api_user'},
            content_type='application/json'
        )
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
