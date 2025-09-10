Run main.py

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
Create Index
Create a var INDEX_NAME=custom-index

