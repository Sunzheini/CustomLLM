"""
Create a bridge between your Python application and the TypeScript MCP weather server,
with a Flask API wrapper
"""
import subprocess
import threading
import time
import json

import requests
import pytest
from flask import Flask, jsonify


class MCPWeatherBridge:
    """Bridge to communicate with the MCP weather server."""
    def __init__(self, server_script):
        self.server_script = server_script
        self.process = None
        self.ready = False
        self.response_queue = []
        self.response_condition = threading.Condition()
        self.available_tools = []
        self.start_server()
        self.start_stdout_reader()
        self.initialize_mcp_session()

    def start_server(self):
        """Start the MCP weather server as a subprocess."""
        print(f"Starting MCP server: {self.server_script}")
        self.process = subprocess.Popen(
            ['node', self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        self.stderr_thread = threading.Thread(target=self._read_stderr)
        self.stderr_thread.daemon = True
        self.stderr_thread.start()

    def _read_stderr(self):
        """Read stderr from the process."""
        while True:
            line = self.process.stderr.readline()
            if line:
                print(f"MCP Server stderr: {line.strip()}")
            else:
                break

    def start_stdout_reader(self):
        """Start a thread to continuously read stdout."""
        self.stdout_thread = threading.Thread(target=self._read_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

    def _read_stdout(self):
        """Continuously read stdout and put messages in queue."""
        while True:
            line = self.process.stdout.readline()
            if line:
                print(f"Received: {line.strip()}")
                try:
                    message = json.loads(line)
                    with self.response_condition:
                        self.response_queue.append(message)
                        self.response_condition.notify_all()
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {line}")
            else:
                break

    def _send_message(self, message):
        """Send a message to the MCP server."""
        message_str = json.dumps(message) + '\n'
        print(f"Sending: {message_str.strip()}")
        self.process.stdin.write(message_str)
        self.process.stdin.flush()

    def _wait_for_response(self, expected_id, timeout=20):
        """Wait for a response with the expected ID."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.response_condition:
                for i, response in enumerate(self.response_queue):
                    if response.get('id') == expected_id:
                        return self.response_queue.pop(i)
                self.response_condition.wait(0.1)
        return None

    def initialize_mcp_session(self):
        """Initialize the MCP session and discover available tools."""
        try:
            time.sleep(2)

            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "WeatherClient",
                        "version": "1.0.0"
                    }
                }
            }

            self._send_message(init_request)
            response = self._wait_for_response(1, timeout=10)

            if response and 'result' in response:
                print("MCP initialization successful")

                # Send initial notification
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }
                self._send_message(notification)

                # List available tools
                self.discover_tools()
                self.ready = True
                print(f"MCP session initialized with tools: {[t['name'] for t in self.available_tools]}")
            else:
                print("MCP initialization failed")
                self.ready = False

        except Exception as e:
            print(f"Failed to initialize MCP session: {e}")
            self.ready = False

    def discover_tools(self):
        """Discover available tools from the server."""
        try:
            request_id = 2
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/list",
                "params": {}
            }

            self._send_message(request)
            response = self._wait_for_response(request_id, timeout=10)

            if response and 'result' in response:
                self.available_tools = response['result'].get('tools', [])
                print(f"Discovered tools: {[tool['name'] for tool in self.available_tools]}")
            else:
                print("No tools discovered or failed to list tools")

        except Exception as e:
            print(f"Failed to discover tools: {e}")

    def get_alerts(self, state_code):
        """Get weather alerts for a state."""
        if not self.ready:
            raise Exception("MCP session not initialized")

        try:
            request_id = int(time.time() * 1000)
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "get-alerts",
                    "arguments": {
                        "state": state_code.upper()
                    }
                }
            }

            self._send_message(request)
            response = self._wait_for_response(request_id, timeout=10)

            if response:
                if 'result' in response:
                    return response['result']
                elif 'error' in response:
                    return f"Error: {response['error']}"
            else:
                return "Error: No response from server"

        except Exception as e:
            print(f"Error getting alerts: {e}")
            return f"Error: {str(e)}"

    def get_forecast(self, latitude, longitude):
        """Get weather forecast for a location."""
        if not self.ready:
            raise Exception("MCP session not initialized")

        try:
            request_id = int(time.time() * 1000)
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {
                    "name": "get-forecast",
                    "arguments": {
                        "latitude": latitude,
                        "longitude": longitude
                    }
                }
            }

            self._send_message(request)
            response = self._wait_for_response(request_id, timeout=10)

            if response:
                if 'result' in response:
                    return response['result']
                elif 'error' in response:
                    return f"Error: {response['error']}"
            else:
                return "Error: No response from server"

        except Exception as e:
            print(f"Error getting forecast: {e}")
            return f"Error: {str(e)}"

    def cleanup(self):
        """Clean up the process."""
        if self.process:
            self.process.terminate()
            self.process.wait()


@pytest.fixture(scope="module")
def weather_server():
    """Fixture to start the MCP weather server for tests."""
    app = Flask(__name__)

    # Initialize the bridge
    try:
        weather_bridge = MCPWeatherBridge(
            r'D:\Study\Projects\Github\mcp_servers\quickstart-resources\weather-server-typescript\build\index.js'
        )
    except Exception as e:
        print(f"Failed to initialize MCP bridge: {e}")
        weather_bridge = None

    @app.route('/alerts/<state_code>')
    def get_alerts(state_code):
        if not weather_bridge or not weather_bridge.ready:
            return jsonify({'error': 'MCP server not ready'}), 503

        try:
            result = weather_bridge.get_alerts(state_code)
            return jsonify({'alerts': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/forecast/<latitude>/<longitude>')
    def get_forecast(latitude, longitude):
        if not weather_bridge or not weather_bridge.ready:
            return jsonify({'error': 'MCP server not ready'}), 503

        try:
            result = weather_bridge.get_forecast(float(latitude), float(longitude))
            return jsonify({'forecast': result})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/health')
    def health_check():
        status = 'ready' if weather_bridge and weather_bridge.ready else 'not ready'
        tools = [t['name'] for t in weather_bridge.available_tools] if weather_bridge else []
        return jsonify({'status': status, 'mcp_server': status, 'available_tools': tools})

    # Start Flask app in a separate thread
    def run_app():
        app.run(port=5001, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Wait for server to start
    time.sleep(3)

    yield app  # This provides the app to tests

    # Cleanup after tests
    if weather_bridge:
        weather_bridge.cleanup()


def test_weather_api(weather_server):
    """Test the weather API using the running server."""
    base_url = "http://localhost:5001"

    print("Testing MCP Weather API with correct tools...")

    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Status: {response.status_code}")
    health_data = response.json()
    print(f"Response: {json.dumps(health_data, indent=2)}")

    assert response.status_code == 200
    assert health_data['status'] == 'ready'

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
                print(f"Alerts response received for {state}")
                assert 'alerts' in alert_data
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")
            pytest.fail(f"Failed to get alerts for {state}: {e}")

    # Test 3: Get forecasts for major cities
    print("\n3. Testing weather forecasts...")
    locations = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
    ]

    for loc in locations:
        print(f"\nTesting forecast for {loc['name']}...")
        try:
            response = requests.get(f"{base_url}/forecast/{loc['lat']}/{loc['lon']}", timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                forecast_data = response.json()
                print(f"Forecast response received for {loc['name']}")
                assert 'forecast' in forecast_data
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")


def test_health_endpoint(weather_server):
    """Test the health endpoint specifically."""
    base_url = "http://localhost:5001"

    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200

    data = response.json()
    assert data['status'] in ['ready', 'not ready']
    assert 'available_tools' in data


if __name__ == "__main__":
    # This allows you to run the test directly for debugging
    pytest.main([__file__, "-v"])
