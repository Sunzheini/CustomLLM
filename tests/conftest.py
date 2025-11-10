import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pinecone import Pinecone

from embeddings.embeddings_manager import EmbeddingsManager
from tools.tools_manager import ToolsManager
from vector_stores.vector_store_manager import VectorStoreManager
from prompts.prompt_manager import PromptManager
from llm_manager import LlmManager
from chains.chains_manager import ChainsManager
from communication.communications_manager import CommunicationsManager


load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


def split_document(file_type, file_path) -> list[Document] | None:
    """
    Loads and splits the file (.pdf or .txt) once for multiple tests.
    """
    if not Path(file_path).exists():
        pytest.skip("File not found - cannot test splitting")
        return None

    if file_type != '.pdf' and file_type != '.txt':
        pytest.skip("Only .pdf and .txt files are supported in this test")
        return None

    loader = None
    if file_type == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == '.txt':
        loader = TextLoader(file_path, encoding='utf8')

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    texts = text_splitter.split_documents(documents)

    print(f"Document has been split into {len(texts)} chunks.")
    return texts


def split_document_list(documents_list) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents_list)
    print(f"Document has been split into {len(texts)} chunks.")
    return texts


async def run_crawl():
    tavily_crawl = TavilyCrawl()

    result = await tavily_crawl.ainvoke({
        'url': "https://python.langchain.com/",
        'max_depth': 2,                             # start with 2, then increase to max 5
        'extract_depth': 'advanced',
        'instructions': 'content on ai agents',     # focus on this topic
    })

    initial_results = result['results']
    print(f"Initial crawl found {len(initial_results)} results")

    # Create Document objects from crawl results; ensure we use the correct item variable and skip empty content
    all_docs = [
        Document(
            page_content=item['raw_content'],
            metadata={"source": item.get('url')}
        ) for item in initial_results if item.get('raw_content')
    ]
    print(f"{len(all_docs)} documents created from initial crawl")

    return all_docs


@pytest.fixture(scope="session")
def base_dir():
    return Path(__file__).resolve().parent.parent


def get_managers():
    return {
        'embeddings_manager': EmbeddingsManager(),              # 0
        'tools_manager': ToolsManager(),                        # 1
        'vector_store_manager': VectorStoreManager(),           # 2
        'prompt_manager': PromptManager(),                      # 3
        'llm_manager': LlmManager(),                            # 4
        'chains_manager': ChainsManager(),                      # 5
        'communications_manager': CommunicationsManager(),      # 6
    }


@pytest.fixture(scope="session")
def managers():
    return get_managers()


@pytest.fixture(scope="session", autouse=True)
def cleanup_records_in_cloud_vector_store_after_tests(base_dir):
    """
    Auto-run fixture that cleans up after all tests are done.
    """
    yield  # This runs before tests

    """Delete all vectors from a Pinecone index."""

    all_indexes_in_pinecone = [index_name, index2_name]
    try:
        if not pinecone_api_key or not all_indexes_in_pinecone:
            print("Warning: Missing Pinecone API key or index names for cleanup")
            return

        # Initialize Pinecone client (v3+ syntax)
        pc = Pinecone(api_key=pinecone_api_key)

        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]

        for idx in all_indexes_in_pinecone:
            if idx not in existing_indexes:
                print(f"Index {idx} does not exist, nothing to clean up there")
                return

        for idx in all_indexes_in_pinecone:
            # Get the index
            index = pc.Index(idx)

            # Get index stats to see if there are vectors to delete
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)

            if total_vectors > 0:
                print(f"Cleaning up {total_vectors} vectors from {index}")

                index.delete(delete_all=True)

                print("✓ Pinecone index cleaned up successfully")
            else:
                print("No vectors to clean up")

    except Exception as e:
        print(f"Warning: Could not cleanup Pinecone indexes {[idx for idx in all_indexes_in_pinecone]}: {e}")
        # Don't fail the tests due to cleanup issues
        pass


# run with `pytest -v` in th terminal
# run with `pytest -v -s` to see print statements
# to debug right click on the test and select "Debug ..."
# pytest - k "llm_query" to execute a test by keyword
