import ollama

"""
Uses the official Ollama Python client to interact with a local Ollama server.
"""


class OllamaAutomatic:
    def __init__(self, model="gemma3:270m", host="http://localhost:11434"):
        """
        Initialize the Ollama client

        Args:
            model: The model name to use
            host: The Ollama server host URL
        """
        self.model = model
        self.host = host
        self.client = ollama.Client(host=host)

    def check_connection(self):
        """Check if Ollama server is running and accessible"""
        try:
            # Try to list models as a connection test
            self.client.list()
            print("Ollama server is running and accessible.")
            return True
        except (ollama.ResponseError, ConnectionError) as e:
            print(f"Failed to connect to Ollama server: {e}")
            return False

    def generate_response(self, prompt, **kwargs):
        """
        Generate a response using the Ollama client

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for generation (e.g., temperature, top_p)

        Returns:
            The generated response or None if failed
        """
        if not self.check_connection():
            print("Ollama server is not running. Please start the server and try again.")
            return None

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                **kwargs
            )
            return response.response

        except ollama.ResponseError as e:
            print(f"Error generating response: {e}")
            return None

    def chat(self, messages, **kwargs):
        """
        Chat with the model using message history

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for generation

        Returns:
            The chat response or None if failed
        """
        if not self.check_connection():
            print("Ollama server is not running. Please start the server and try again.")
            return None

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                **kwargs
            )
            return response.message.content

        except ollama.ResponseError as e:
            print(f"Error in chat: {e}")
            return None

    def stream_generate(self, prompt, **kwargs):
        """
        Generate a response with streaming

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for generation

        Yields:
            Response chunks as they are generated
        """
        if not self.check_connection():
            print("Ollama server is not running. Please start the server and try again.")
            return

        try:
            stream = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                **kwargs
            )

            print("Streaming response from Ollama:")
            full_response = ""
            for chunk in stream:
                if chunk.response:
                    print(chunk.response, end="", flush=True)
                    full_response += chunk.response
                    yield chunk.response

            print()  # Final newline
            yield full_response  # Final yield with complete response

        except ollama.ResponseError as e:
            print(f"Error in streaming generation: {e}")
            yield None
