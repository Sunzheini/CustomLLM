import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preparation():
    """
    This function prepares the vector store for semantic search by loading documents,
    :return: The Ollama embeddings, the database location, and a flag indicating whether to add documents.
    """
    # converts text data into embeddings using the Ollama model to enable semantic search
    prepared_embeddings = OllamaEmbeddings(model="codellama")

    # initializes a Chroma vector store to store and retrieve documents based on their embeddings
    prepared_db_location = "./chrome_langchain_db"

    # checks if the database already exists; if not, it will add documents to the vector store
    add_documents = not os.path.exists(prepared_db_location)

    # initializes a Chroma vector store with the specified collection name and persistence directory
    vector_store = Chroma(
        collection_name="document_collection",
        persist_directory=prepared_db_location,
        embedding_function=prepared_embeddings
    )

    return add_documents, vector_store


def fill_info(add_documents, vector_store, data_type: str):
    """
    This function fills the vector store with documents from a specified directory, and prepares them for semantic search.
    THen it creates a retriever to fetch relevant documents based on a query.
    :param add_documents: Flag indicating whether to add documents to the vector store.
    :param vector_store: The Chroma vector store instance where documents will be stored.
    :param data_type: The type of data to be loaded, either "scv" for CSV files or "pdf_folder" for PDF files.
    :return: A retriever that can be used to retrieve relevant documents based on a query.
    """
    if add_documents:
        if data_type == "scv":
            # creates a pandas DataFrame from a CSV file containing restaurant reviews
            # pandas_data_frame = pd.read_csv("./context/realistic_restaurant_reviews.csv")
            pandas_data_frame = pd.read_excel("./context/sample_context_data.xlsx")

            # if the database does not exist, it will add documents to the vector store
            final_docs = []
            ids = []

            for i, row in pandas_data_frame.iterrows():
                document = Document(
                    # Create a Document object for each review
                    # page_content=row["Title"] + " " + row["Review"],
                    page_content=row["Day"] + " " + row["Routine"],
                    # metadata={"rating": row["Rating"], "date": row["Date"]},
                    metadata={
                        "date": row["Date"]
                    },
                    id=str(i)
                )
                ids.append(str(i))
                final_docs.append(document)

        if data_type == "pdf_folder":
            # Load all PDFs from directory (returns one Document per file)
            loader = DirectoryLoader(
                './context/',
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True  # Faster loading for multiple files
            )
            raw_documents = loader.load()

            # Configure text splitter for optimal chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Size of each chunk (in characters)
                chunk_overlap=200,  # Overlap between chunks
                length_function=len,  # How to measure chunk size
                is_separator_regex=False,  # Treat separators as literal strings
                separators=["\n\n", "\n", " ", ""]  # Splitting hierarchy
            )

            # Split documents into chunks
            documents = text_splitter.split_documents(raw_documents)

            # Prepare documents with proper IDs
            final_docs = []
            ids = []
            for i, doc in enumerate(documents):
                final_docs.append(Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"chunk_{i}",  # Add chunk identifier
                        "total_chunks": len(documents)  # For reference
                    }
                ))
                ids.append(str(i))

        # Add documents to the vector store
        vector_store.add_documents(documents=final_docs, ids=ids)

    # creates a retriever from the vector store to retrieve relevant documents based on a query
    retriever = vector_store.as_retriever(
        # search_kwargs={"k": 10}     # fetches the top 10 relevant documents for each query
        search_kwargs={
            "k": 1,
        }  # fetches the top 10 relevant documents for each query
    )

    return retriever
