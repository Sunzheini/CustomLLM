Run main.py

# General
Hugging Face = The models (the "what")
LangChain = The orchestration (the "how")
LangSmith = The monitoring (the "why")
LangGraph = orchestration + monitoring
ToDo: use CopilotKit for React

# Ollama
cmd: `ollama`
user: Sunzheini, email: daniel_zorov@abv.bg, pass: Maimun06

`ollama list`           # list installed models

ollama models: https://ollama.com/library
`ollama pull gemma3:270m`       # gemma3:270m is the model
`ollama run gemma3:270m`        # run the model

`ollama ps`                     # list running models
`ollama stop gemma3:270m`       # stop a running model

# OpenAI
models: https://platform.openai.com/settings/organization/limits
usage: https://platform.openai.com/settings/organization/usage

# Vector DB
pip install langchain-pinecone
Pinecone: https://www.pinecone.io/
Create Index:
    e.g. text-embedding-3-small (matching the embeddings in the code), 1536, serverless, aws, default region
Create a var INDEX_NAME=custom-index

# MCP
run `npx @modelcontextprotocol/inspector` -> browser opens
`cd D:\Study\Projects\Github\mcpdoc`
`.venv\Scripts\activate`
`uvx --from mcpdoc mcpdoc --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" "LangChain:https://python.langchain.com/llms.txt" --transport sse --port 8082 --host localhost`
server starts on open http://localhost:8082/
In the MCP Inspector, select transport: SSE, URL: http://localhost:8082/sse, click Connect
