# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend

# Option 1. run with the streamlit runner configuration
backend = ExampleBackend()
frontend = StreamlitFrontend(backend)  # runs in a loop

# Option 2. run a service for AegisAI: use the uvicorn.exe configuration to run
