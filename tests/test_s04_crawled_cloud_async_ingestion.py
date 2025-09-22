import os
import ssl
import asyncio
import certifi
from typing import List
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

from tests.conftest import run_crawl, split_document_list


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index2_name = os.getenv('INDEX2_NAME')


"""
Secure HTTPS Connections - It ensures that your LangChain application can 
securely connect to external APIs and services (like OpenAI, Hugging Face, 
etc.) without SSL certificate verification errors. Avoiding SSL: 
CERTIFICATE_VERIFY_FAILED errors
"""
ssl_context = ssl.create_default_context(
    cafile=certifi.where())  # Creates SSL context using certifi's certificate bundle
os.environ['SSL_CERT_FILE'] = certifi.where()  # Sets environment variable for SSL certificates
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()  # Specifically configures the requests library to use certifi


def test_09_run_crawl_and_split_crawled_documents():
    """
    Test the crawling of documents and their splitting into manageable chunks.
    Ensures that documents are crawled and split correctly with appropriate metadata.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    documents = asyncio.run(run_crawl())

    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    texts = split_document_list(documents)

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    assert len(texts) > 0
    assert isinstance(texts[0], Document)
    assert texts[0].page_content is not None
    assert texts[0].metadata is not None
    assert 2000 <= len(texts[0].page_content) <= 6000  # chunk size


@pytest.mark.asyncio
async def test_10_crawled_cloud_async_ingestion(base_dir, managers):
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    documents = await run_crawl()
    texts = split_document_list(documents)

    # create batches
    texts = [texts[i:i + 10] for i in range(0, len(texts), 10)]

    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings(
        model="text-embedding-3-small",     # one of OpenAI's latest and most cost-effective embedding models
        show_progress_bar=True,             # display a progress bar
        chunk_size=50,                      # how many text strings are sent in a single batch request to the OpenAI API
        retry_min_seconds=10,               # wait at least 10 seconds before retrying the failed request
    )

    # 2
    pinecone_index_name = index2_name
    vectorstore = (managers['vector_store_manager']
    .get_vector_store(
        'pinecone', 'load',
        index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
    ))

    # ASYNCHRONOUS SOLUTION: Controlled concurrency with session recreation and retry logic
    def get_fresh_vectorstore():
        """Create a new vectorstore instance to handle session issues"""
        return (managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'load',
            index_name=pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=pinecone_api_key
        ))

    async def add_batch_with_retry(batch_to_add: List[Document], batch_id: int, semaphore: asyncio.Semaphore, max_retries: int = 3) -> dict:
        current_vectorstore = vectorstore  # Start with the original

        async with semaphore:  # Limit concurrent requests
            for attempt in range(max_retries):
                try:
                    # Add some jitter to prevent thundering herd
                    if batch_id > 0:
                        await asyncio.sleep(0.1 * batch_id)

                    await current_vectorstore.aadd_documents(batch_to_add)
                    print(
                        f"✓ Batch {batch_id + 1} ({len(batch_to_add)} docs) added successfully (attempt {attempt + 1})")
                    return {"batch_id": batch_id, "success": True, "attempt": attempt + 1}

                except Exception as e:
                    error_msg = str(e).lower()
                    print(f"⚠ Batch {batch_id + 1} error (attempt {attempt + 1}): {e}")

                    # Handle specific error types
                    if "session is closed" in error_msg or "connection" in error_msg or "client session" in error_msg:
                        # For connection issues, recreate vectorstore and wait longer
                        if attempt < max_retries - 1:
                            print(f"🔄 Recreating vectorstore connection for batch {batch_id + 1}...")
                            try:
                                current_vectorstore = get_fresh_vectorstore()
                                print(f"✅ New vectorstore created for batch {batch_id + 1}")
                            except Exception as vs_error:
                                print(f"❌ Failed to recreate vectorstore: {vs_error}")

                            wait_time = (attempt + 1) * 2  # Wait before retry
                            print(f"Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                        continue

                    elif "rate limit" in error_msg or "429" in error_msg:
                        # For rate limiting, wait longer
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 5
                            print(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                            await asyncio.sleep(wait_time)
                        continue

                    else:
                        # For other errors, standard backoff
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            print(f"Retrying batch {batch_id + 1} in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                        continue

            print(f"❌ Batch {batch_id + 1} failed after {max_retries} attempts")
            return {"batch_id": batch_id, "success": False, "attempt": max_retries}

    # Create semaphore to control concurrency (reduced to 2 for better stability)
    max_concurrent = 4  # Reduced from 3 for better session management
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks with batch IDs for better tracking
    add_tasks = [
        add_batch_with_retry(batch, i, semaphore)
        for i, batch in enumerate(texts)
    ]

    print(f"Adding {len(add_tasks)} batches asynchronously with max {max_concurrent} concurrent requests...")
    print("Starting async batch processing...")

    # Execute all tasks concurrently but with controlled concurrency
    results = await asyncio.gather(*add_tasks, return_exceptions=True)

    # Analyze results
    successful_batches = 0
    failed_batches = 0
    total_attempts = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Task failed with exception: {result}")
            failed_batches += 1
        elif isinstance(result, dict):
            if result["success"]:
                successful_batches += 1
            else:
                failed_batches += 1
            total_attempts += result["attempt"]
        else:
            print(f"⚠ Unexpected result type: {result}")
            failed_batches += 1

    avg_attempts = total_attempts / len(results) if results else 0

    print(f"\n📊 ASYNC PROCESSING SUMMARY:")
    print(f"✅ Successful batches: {successful_batches}")
    print(f"❌ Failed batches: {failed_batches}")
    print(f"📈 Average attempts per batch: {avg_attempts:.1f}")
    print(f"🔄 Max concurrent requests: {max_concurrent}")

    # ----------------------------------------------------------------------------------
    # Assert - Verify the vector store was created correctly
    # ----------------------------------------------------------------------------------
    assert successful_batches == len(texts)
