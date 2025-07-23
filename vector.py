from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredFileLoader
)


# converts text data into embeddings using the Ollama model to enable semantic search
embeddings = OllamaEmbeddings(model="codellama")

# initializes a Chroma vector store to store and retrieve documents based on their embeddings
db_location = "./chrome_langchain_db"

# checks if the database already exists; if not, it will add documents to the vector store
add_documents = not os.path.exists(db_location)

# ----------------------------------------------------------------------

# creates a pandas DataFrame from a CSV file containing restaurant reviews
# pandas_data_frame = pd.read_csv("./context/realistic_restaurant_reviews.csv")
# pandas_data_frame = pd.read_excel("./context/sample_context_data.xlsx")
#
# # if the database does not exist, it will add documents to the vector store
# if add_documents:
#     documents = []
#     ids = []
#
#     for i, row in pandas_data_frame.iterrows():
#         document = Document(
#             # Create a Document object for each review
#             # page_content=row["Title"] + " " + row["Review"],
#             page_content=row["Day"] + " " + row["Routine"],
#             # metadata={"rating": row["Rating"], "date": row["Date"]},
#             metadata={
#                 "date": row["Date"]
#             },
#             id=str(i)
#         )
#         ids.append(str(i))
#         documents.append(document)

# Load PDF documents if database doesn't exist
if add_documents:
    # Initialize PDF loader - point this to your PDF file or directory
    loader = PyPDFLoader("./context/CV_DZ.pdf")  # Change this path

    # Load and split pages (each page becomes a document)
    pages = loader.load_and_split()

    # Prepare documents for Chroma
    documents = []
    ids = []

    for i, page in enumerate(pages):
        document = Document(
            page_content=page.page_content,
            metadata={
                "source": page.metadata.get("source", ""),
                "page": page.metadata.get("page", 0),
                # Add any other metadata you want to preserve
            }
        )
        ids.append(str(i))
        documents.append(document)

# ----------------------------------------------------------------------

# initializes a Chroma vector store with the specified collection name and persistence directory
vector_store = Chroma(
    collection_name="document_collection",
    persist_directory=db_location,
    embedding_function=embeddings
)

# if the database does not exist, it will add the documents to the vector store
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# creates a retriever from the vector store to retrieve relevant documents based on a query
retriever = vector_store.as_retriever(
    # search_kwargs={"k": 10}     # fetches the top 10 relevant documents for each query
    search_kwargs={"k": 1}     # fetches the top 10 relevant documents for each query
)
