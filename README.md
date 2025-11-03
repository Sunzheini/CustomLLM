# CustomLLM
An app that demonstrates many use cases of LLMs with LangChain, Hugging Face, 
Ollama, Pinecone, MCP and more. You can check the tests for the various use cases.
Also there are 2 concrete implementations: main py and when running the uvicorn configuration


## Run main.py
### General Info
Hugging Face = The models (the "what")
LangChain = The orchestration (the "how")
LangSmith = The monitoring (the "why")
LangGraph = orchestration + monitoring
ToDo: use CopilotKit for React

### Streamlit runner
Script: D:/Study/Projects/Github/CustomLLM/.venv/Scripts/streamlit.exe
Parameters: run main.py
Working directory: D:/Study/Projects/Github/CustomLLM
Path to env files: D:/Study/Projects/Github/CustomLLM/.env

### Ollama
cmd: `ollama`
user: Sunzheini, email: daniel_zorov@abv.bg, pass: Maimun06

`ollama list`           # list installed models

ollama models: https://ollama.com/library
`ollama pull gemma3:270m`       # gemma3:270m is the model
`ollama run gemma3:270m`        # run the model

`ollama ps`                     # list running models
`ollama stop gemma3:270m`       # stop a running model

### OpenAI
models: https://platform.openai.com/settings/organization/limits
usage: https://platform.openai.com/settings/organization/usage

### Vector DB
pip install langchain-pinecone
Pinecone: https://www.pinecone.io/
Create Index:
    e.g. text-embedding-3-small (matching the embeddings in the code), 1536, serverless, aws, default region
Create a var INDEX_NAME=custom-index

### MCP
example: 
run `npx @modelcontextprotocol/inspector` -> browser opens
`cd D:\Study\Projects\Github\mcpdoc`
`.venv\Scripts\activate`
`uvx --from mcpdoc mcpdoc --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" "LangChain:https://python.langchain.com/llms.txt" --transport sse --port 8082 --host localhost`
server starts on open http://localhost:8082/
In the MCP Inspector, select transport: SSE, URL: http://localhost:8082/sse, click Connect

run a local mcp server to use it inside your app:
1. clone `https://github.com/modelcontextprotocol/quickstart-resources` inside `D:\Study\Projects\Github\mcp_servers
2. have npm and `npm install -g typescript`
3. `cd D:\Study\Projects\Github\mcp_servers\quickstart-resources\weather-server-typescript\` and run `npm install`


## If using the app as a service for AegisAI, run the configuration 
1. Clone the repository
2. Have poetry installed in your default python environment
3. Put  `"shared-lib @ file:///C:/Workspace/Python/AegisAI/shared-lib",` in the requirements.txt of CustomLLM
4. You need an .env file in the root of CustomLLM
5. Change to your dir, e.g.`
STORAGE_ROOT=C:/Workspace/Python/AegisAI/shared-storage
RAW_DIR=C:/Workspace/Python/AegisAI/shared-storage/raw
PROCESSED_DIR=C:/Workspace/Python/AegisAI/shared-storage/processed
TRANSCODED_DIR=C:/Workspace/Python/AegisAI/shared-storage/transcoded
` in the .env
6. Delete venvs if i have pushed the by mistake..
7. Edit requirements.txt, replace:
`
opentelemetry-api==1.36.0
opentelemetry-exporter-otlp-proto-common==1.36.0
opentelemetry-exporter-otlp-proto-grpc==1.36.0
opentelemetry-instrumentation==0.53b1
opentelemetry-instrumentation-asgi==0.53b1
opentelemetry-instrumentation-fastapi==0.53b1
opentelemetry-proto==1.36.0
opentelemetry-sdk==1.36.0
opentelemetry-semantic-conventions==0.57b0
opentelemetry-util-http==0.53b1
`
with:
`
opentelemetry-api==1.38.0
opentelemetry-exporter-otlp-proto-common==1.38.0
opentelemetry-exporter-otlp-proto-grpc==1.38.0
opentelemetry-instrumentation==0.59b0
opentelemetry-instrumentation-asgi==0.59b0
opentelemetry-instrumentation-fastapi==0.59b0
opentelemetry-proto==1.38.0
opentelemetry-sdk==1.38.0
opentelemetry-semantic-conventions==0.59b0
opentelemetry-util-http==0.59b0
`
8. You must have Build Tools for Visual Studio 2022:
Open the Visual Studio Installer → find Build Tools for Visual Studio 2022 → click Modify → ensure these boxes are checked:
    Desktop development with C++
    Under “Installation details” on the right:
	    MSVC v143 - VS 2022 C++ x64/x86 build tools
	    Windows 10 or 11 SDK
	    C++ CMake tools for Windows
Install and Restart
Press Start → type “Developer Command Prompt for VS 2022”.
Right-click → Run as administrator (optional, but helps avoid permission issues).
In that window, type: cl
You should now see something like: Microsoft (R) C/C++ Optimizing Compiler Version 19.3x for x64
9. Open a terminal inside the project and run:
`python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements.txt`
10. Open the project
11. As interpreter select .venv/Scripts/python.exe