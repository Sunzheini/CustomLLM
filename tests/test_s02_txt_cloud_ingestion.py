import os
from pathlib import Path

import pytest
from pinecone import Pinecone
from dotenv import load_dotenv

from tests.conftest import split_pdf


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')


def test_04_split_txt_into_chunks():
    """
    Test splitting a txt document into text chunks.
    """
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    path_to_file = './context/exampleblog.txt'
    texts = split_pdf('.txt', path_to_file)

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    assert len(texts) > 0, "Should have at least one text chunk"
    assert all(len(text.page_content) > 0 for text in texts), "All chunks should have content"
    assert all(hasattr(text, 'metadata') for text in texts), "All chunks should have metadata"

    # Print some sample info
    print(f"First chunk length: {len(texts[0].page_content)} characters")
    print(f"Sample chunk: {texts[0].page_content[:100]}...")


def test_05_ingest_txt_into_cloud_vector_store(base_dir, managers):
    """
    Test ingesting txt content into a vector store and querying it.
    This is an integration test that requires OpenAI API access.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    path_to_file = './context/exampleblog.txt'
    texts = split_pdf('.txt', path_to_file)

    # reduce texts to 3 for faster testing
    texts = texts[:3]

    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings

    # 2
    pinecone_index_name = index_name
    vectorstore = (managers['vector_store_manager']
                        .get_vector_store(
                            'pinecone', 'create',
                            texts, embeddings,
                            index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key
                        ))

    # ----------------------------------------------------------------------------------
    # Assert - Verify the vector store was created correctly
    # ----------------------------------------------------------------------------------
    assert vectorstore is not None, "Vector store should be created"
    assert hasattr(vectorstore, 'as_retriever'), "Vector store should have retriever method"

    # Test basic retrieval functionality
    retriever = vectorstore.as_retriever()
    assert retriever is not None, "Should be able to create a retriever"

    # Test that documents were actually ingested by doing a simple search
    try:
        # Search for something that should exist in your text
        results = vectorstore.similarity_search("Wiki", k=1)
        assert len(results) > 0, "Should find at least one similar document"
        assert hasattr(results[0], 'page_content'), "Results should have content"
        assert len(results[0].page_content) > 0, "Result content should not be empty"

        print(f"Successfully retrieved document: {results[0].page_content[:100]}...")

    except Exception as e:
        pytest.fail(f"Failed to query Pinecone index: {e}")


@pytest.fixture(scope="session", autouse=True)
def cleanup_records_in_cloud_vector_store_after_tests(base_dir):
    """
    Auto-run fixture that cleans up after all tests are done.
    """
    yield  # This runs before tests

    """Delete all vectors from a Pinecone index."""
    try:
        if not pinecone_api_key or not index_name:
            print("Warning: Missing Pinecone API key or index name for cleanup")
            return

        # Initialize Pinecone client (v3+ syntax)
        pc = Pinecone(api_key=pinecone_api_key)

        # Check if index exists
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            print(f"Index {index_name} does not exist, nothing to clean up")
            return

        # Get the index
        index = pc.Index(index_name)

        # Get index stats to see if there are vectors to delete
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)

        if total_vectors > 0:
            print(f"Cleaning up {total_vectors} vectors from {index_name}")

            # Option 1: Delete all vectors from all namespaces
            index.delete(delete_all=True)

            # Option 2: If you need to delete from specific namespaces:
            # namespaces = stats.get('namespaces', {})
            # for namespace in namespaces.keys():
            #     print(f"Deleting vectors from namespace: {namespace}")
            #     index.delete(delete_all=True, namespace=namespace)

            print("✓ Pinecone index cleaned up successfully")
        else:
            print("No vectors to clean up")

    except Exception as e:
        print(f"Warning: Could not cleanup Pinecone index {index_name}: {e}")
        # Don't fail the tests due to cleanup issues
        pass
