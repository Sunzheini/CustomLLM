# test_weather_fixed.py
import requests
import json


def test_weather_api():
    base_url = "http://localhost:5001"

    print("Testing MCP Weather API with correct tools...")

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    health_data = response.json()
    print(f"Response: {json.dumps(health_data, indent=2)}")

    # Test 2: Get alerts for states
    print("\n2. Testing weather alerts...")
    states = ["CA", "NY", "FL", "TX"]

    for state in states:
        print(f"\nTesting alerts for {state}...")
        try:
            response = requests.get(f"{base_url}/alerts/{state}", timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                alert_data = response.json()
                print(f"Alerts response: {json.dumps(alert_data, indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

    # Test 3: Get forecasts for major cities
    print("\n3. Testing weather forecasts...")
    locations = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Paris", "lat": 48.8566, "lon": 2.3522}
    ]

    for loc in locations:
        print(f"\nTesting forecast for {loc['name']}...")
        try:
            response = requests.get(f"{base_url}/forecast/{loc['lat']}/{loc['lon']}", timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                forecast_data = response.json()
                print(f"Forecast response: {json.dumps(forecast_data, indent=2)}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")


if __name__ == "__main__":
    test_weather_api()