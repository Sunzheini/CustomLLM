from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_tavily import TavilyCrawl
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from embeddings.embeddings_manager import EmbeddingsManager
from tools.tools_manager import ToolsManager
from vector_stores.vector_store_manager import VectorStoreManager
from prompts.prompt_manager import PromptManager
from llm_manager import LlmManager
from chains.chains_manager import ChainsManager
from communication.communications_manager import CommunicationsManager


def split_document(file_type, file_path):
    """
    Loads and splits the file (.pdf or .txt) once for multiple tests.
    """
    if not Path(file_path).exists():
        pytest.skip("File not found - cannot test splitting")
        return None

    if file_type != '.pdf' and file_type != '.txt':
        pytest.skip("Only .pdf and .txt files are supported in this test")
        return None

    if file_type == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == '.txt':
        loader = TextLoader(file_path, encoding='utf8')

    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    texts = text_splitter.split_documents(documents)

    print(f"Document has been split into {len(texts)} chunks.")
    return texts


def split_document_list(documents_list):
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

    all_docs = [
        Document(
            page_content=result['raw_content'],
            metadata={"source": result['url']}
        ) for result in result['results']]
    print(f"{len(all_docs)} documents created from initial crawl")

    return all_docs


@pytest.fixture(scope="session")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def managers():
    return {
        'embeddings_manager': EmbeddingsManager(),              # 0
        'tools_manager': ToolsManager(),                        # 1
        'vector_store_manager': VectorStoreManager(),           # 2
        'prompt_manager': PromptManager(),                      # 3
        'llm_manager': LlmManager(),                            # 4
        'chains_manager': ChainsManager(),                      # 5
        'communications_manager': CommunicationsManager(),      # 6
    }


# run with `pytest -v` in th terminal
# run with `pytest -v -s` to see print statements
# to debug right click on the test and select "Debug ..."
# pytest - k "llm_query" to execute a test by keyword
