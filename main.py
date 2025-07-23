from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


# This script uses the Ollama LLM to answer questions
model = OllamaLLM(model="codellama")

# The prompt template to format the questions and context for the LLM
template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

# Create a chat prompt template using the defined template
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and model into a chain to process the input
chain = prompt | model

# This script allows you to ask questions and get answers based on the documents stored in the vector database.
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    # Retrieve relevant documents from the vector store based on the question
    reviews = retriever.invoke(question)

    # Invoke the chain with the retrieved documents and the question to get an answer
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)