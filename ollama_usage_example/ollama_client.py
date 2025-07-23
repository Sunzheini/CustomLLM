import ollama


client = ollama.Client()

model = "codellama"  # Replace with the model name you're using
prompt = "Explain python dict in 1 sentence"  # Replace with your input prompt

response = client.generate(
    model=model,
    prompt=prompt,
)

print("Response from Ollama:")
print(response.response)
