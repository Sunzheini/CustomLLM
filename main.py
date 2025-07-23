from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


# This script uses the Ollama LLM to answer questions
model = OllamaLLM(model="codellama")

template = """
You are an expert in {domain}. Answer questions based on the provided context.

Context: {reviews}

Question: {question}
"""

# summarize work experience
domain = "human resources"

# Create a chat prompt template using the defined template
prompt = ChatPromptTemplate.from_template(template)

# Combine the prompt and model into a chain to process the input
chain = {"reviews": lambda x: x["reviews"],
         "question": lambda x: x["question"],
         "domain": lambda _: domain} | prompt | model

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
