import shutil
from pathlib import Path

import pytest
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


@pytest.fixture(scope="session")
def split_pdf_texts():
    """
    Fixture that loads and splits the PDF once for multiple tests.
    Returns the split texts or None if PDF not found.
    """
    path_to_file = './context/react_paper.pdf'

    # Check if file exists first
    if not Path(path_to_file).exists():
        pytest.skip("PDF file not found - cannot test splitting")
        return None

    loader = PyPDFLoader(path_to_file)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    texts = text_splitter.split_documents(documents)

    print(f"Document has been split into {len(texts)} chunks.")
    return texts


def test_01_split_pdf_into_chunks(split_pdf_texts):
    """
    Test splitting a PDF document into text chunks.
    Verifies that the document is loaded and split correctly.
    """
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    texts = split_pdf_texts

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    assert len(texts) > 0, "Should have at least one text chunk"
    assert all(len(text.page_content) > 0 for text in texts), "All chunks should have content"
    assert all(hasattr(text, 'metadata') for text in texts), "All chunks should have metadata"

    # Print some sample info
    print(f"First chunk length: {len(texts[0].page_content)} characters")
    print(f"Sample chunk: {texts[0].page_content[:100]}...")


def test_02_ingest_pdf_to_vector_store(base_dir, managers, split_pdf_texts):
    """
    Test ingesting PDF content into a vector store and querying it.
    This is an integration test that requires OpenAI API access.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    texts = split_pdf_texts

    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings

    # 2
    faiss_path = str(base_dir / 'faiss_index_react_paper')
    vectorstore = managers['vector_store_manager'].get_vector_store('faiss', 'create', texts, embeddings)

    vectorstore.save_local('faiss_index_react_paper')    # if you want to save to disc, otherwise it is in memory only
    print(f"Vector store saved to: {faiss_path}")

    # ----------------------------------------------------------------------------------
    # Assert - Verify the vector store was created correctly
    # ----------------------------------------------------------------------------------
    assert vectorstore is not None, "Vector store should be created"
    assert hasattr(vectorstore, 'as_retriever'), "Vector store should have retriever method"


@pytest.fixture(scope="session", autouse=True)
def cleanup_vector_store_after_tests(base_dir):
    """
    Auto-run fixture that cleans up after all tests are done.
    """
    yield  # This runs before tests

    # This runs AFTER all tests complete
    faiss_path = base_dir / 'faiss_index_react_paper'
    if faiss_path.exists():
        shutil.rmtree(faiss_path)
        print(f"Cleaned up test vector store: {faiss_path}")
