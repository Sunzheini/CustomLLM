from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


# creates a pandas DataFrame from a CSV file containing restaurant reviews
pandas_data_frame = pd.read_csv("realistic_restaurant_reviews.csv")

# converts text data into embeddings using the Ollama model to enable semantic search
embeddings = OllamaEmbeddings(model="codellama")

# initializes a Chroma vector store to store and retrieve documents based on their embeddings
db_location = "./chrome_langchain_db"

# checks if the database already exists; if not, it will add documents to the vector store
add_documents = not os.path.exists(db_location)

# if the database does not exist, it will add documents to the vector store
if add_documents:
    documents = []
    ids = []

    for i, row in pandas_data_frame.iterrows():
        document = Document(
            # Create a Document object for each review
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

# initializes a Chroma vector store with the specified collection name and persistence directory
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# if the database does not exist, it will add the documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# creates a retriever from the vector store to retrieve relevant documents based on a query
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}     # fetches the top 10 relevant documents for each query
)
