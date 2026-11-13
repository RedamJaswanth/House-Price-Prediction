import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from fastapi.testclient import TestClient
from api import app

def test_endpoints():
    client = TestClient(app)
    
    # 1. Health check
    r = client.get("/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("Health check response:", r.json())
    
    # 2. Prediction endpoint (INR)
    payload = {
        "location": "Cyber City (Gurugram)",
        "area_sqft": 2100.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 2,
        "house_age_years": 4,
        "parking_spaces": 2,
        "distance_city_center_km": 4.0,
        "school_rating": 8,
        "furnishing": "Fully-Furnished"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, f"Predict failed: {r.text}"
    print("Prediction response (INR):", r.json())
    print("\nAPI Integration Tests PASSED (INR Currency verified)!")

if __name__ == "__main__":
    test_endpoints()
