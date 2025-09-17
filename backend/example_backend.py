import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone

from support.callback_handler import CustomCallbackHandler
from tests.conftest import run_crawl, split_document_list, get_managers


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


class ExampleBackend:
    def __init__(self):
        self.managers = get_managers()

    async def crawled_cloud_async_ingestion(self):
        documents = await run_crawl()
        texts = split_document_list(documents)

        # create batches
        texts = [texts[i:i + 10] for i in range(0, len(texts), 10)]

        # 0
        embeddings = self.managers['embeddings_manager'].open_ai_embeddings(
            model="text-embedding-3-small",
            show_progress_bar=True,
            chunk_size=50,
            retry_min_seconds=10,
        )

        # 2
        pinecone_index_name = index2_name
        vectorstore = (self.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'load',
            index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
        ))

        # ASYNCHRONOUS SOLUTION: Controlled concurrency with session recreation and retry logic
        def get_fresh_vectorstore():
            """Create a new vectorstore instance to handle session issues"""
            return (self.managers['vector_store_manager']
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

    @staticmethod
    def cleanup_data():
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

    def generate_response(self, user_prompt, chat_history: List[Dict[str, Any]]):
        embeddings = self.managers['embeddings_manager'].open_ai_embeddings(
            model="text-embedding-3-small",
            show_progress_bar=True,
            chunk_size=50,
            retry_min_seconds=10,
        )

        # 2
        pinecone_index_name = index2_name
        vectorstore = (self.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'load',
            index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
        ))

        # 3
        query = user_prompt
        qa_chat_prompt = self.managers['prompt_manager'].get_prompt_template("langchain-ai/retrieval-qa-chat")
        rephrase_prompt = self.managers['prompt_manager'].get_prompt_template("langchain-ai/chat-langchain-rephrase")

        # 4
        llm = self.managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

        # 5
        chain = self.managers['chains_manager'].get_document_retrieval_chain_with_history(
            llm,
            qa_chat_prompt,
            rephrase_prompt,
            vectorstore
        )

        # 6
        response = chain.invoke(input={"input": query, "chat_history": chat_history})

        new_result = {
            "query": response['input'],
            "result": response['answer'],
            "source_documents": response['context'],
        }

        return new_result
