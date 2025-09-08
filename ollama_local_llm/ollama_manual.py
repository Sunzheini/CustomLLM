import requests
import json


class OllamaManual:
    def __init__(self, model="gemma3:270m"):
        self.model = model
        self.ollama_local_server_url = "http://localhost:11434"
        self.ollama_local_server_api_url = f"{self.ollama_local_server_url}/api/chat"

    def check_connection(self):
        """Check if Ollama server is running and return boolean"""
        try:
            response = requests.get(self.ollama_local_server_url, timeout=5)
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def send_request(self, messages):
        """
        Send a request to Ollama and stream the response

        Args:
            messages: List of message dictionaries with 'role' and 'content'
        """
        if not self.check_connection():
            print("Ollama server is not running. Please start the server and try again.")
            return False

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        try:
            response = requests.post(self.ollama_local_server_api_url, json=payload, stream=True)

            if response.status_code == 200:
                print("Streaming response from Ollama:")
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            json_data = json.loads(line)
                            if "message" in json_data and "content" in json_data["message"]:
                                content = json_data["message"]["content"]
                                print(content, end="")
                                full_response += content
                        except json.JSONDecodeError:
                            print(f"\nFailed to parse line: {line}")
                print()  # Final newline
                return full_response
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
