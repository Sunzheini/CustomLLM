import ollama


client = ollama.Client()

model = "codellama"  # Replace with the model name you're using
prompt = "What is 1 + 1?"  # Replace with your input prompt

response = client.generate(
    model=model,
    prompt=prompt,
)

print("Response from Ollama:")
print(response.response)
