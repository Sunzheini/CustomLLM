from backend.example_backend import ExampleBackend
from frontend.example_streamlit_frontend import StreamlitFrontend


backend = ExampleBackend()
frontend = StreamlitFrontend(backend)  # runs in a loop
