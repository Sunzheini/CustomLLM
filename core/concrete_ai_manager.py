"""
Concrete AI implementation
"""
import os
import sys
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone
from shared_lib.contracts.job_schemas import WorkflowGraphState

from support.callback_handler import CustomCallbackHandler
from tests.conftest import split_document, get_managers

load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX_NAME')


class ConcreteAIManager:
    """Concrete AI Manager for handling AI-related operations."""
    base_dir = Path(__file__).resolve().parent.parent
    managers = get_managers()

    @staticmethod
    def split_txt_into_chunks(state: WorkflowGraphState) -> list:
        """
        Test splitting a txt document into text chunks.
        """
        # Check if we have extracted text in metadata
        if state.get("metadata") and state["metadata"].get("text_extraction"):

            # Get the path to the extracted text file
            path_to_file = state["metadata"]["text_extraction"].get("text_file_path")
            if path_to_file and os.path.exists(path_to_file):

                # Split the document into chunks
                texts = split_document('.txt', path_to_file)
                return texts

        return []

    @staticmethod
    def ingest_txt_into_cloud_vector_store(texts: list) -> None:
        """
        Test ingesting txt content into a vector store and querying it.
        This is an integration test that requires OpenAI API access.
        """
        # 0
        embeddings = ConcreteAIManager.managers['embeddings_manager'].open_ai_embeddings()

        # 2
        pinecone_index_name = index_name
        vectorstore = (ConcreteAIManager.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'create',
            texts, embeddings,
            index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key
        ))

    @staticmethod
    def get_retrieval_chain():
        """Create a document retrieval chain with chat history similar to the test cases."""
        # 0
        embeddings = ConcreteAIManager.managers['embeddings_manager'].open_ai_embeddings()

        # 2
        pinecone_index_name = index_name
        vectorstore = (ConcreteAIManager.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'load',
            index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
        ))

        # 3
        retrieval_qa_chat_prompt = ConcreteAIManager.managers['prompt_manager'].get_prompt_template(
            "langchain-ai/retrieval-qa-chat")
        rephrase_prompt = ConcreteAIManager.managers['prompt_manager'].get_prompt_template(
            "langchain-ai/chat-langchain-rephrase")

        # 4
        llm = ConcreteAIManager.managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0,
                                                                callbacks=[CustomCallbackHandler()])

        # 5
        chain = ConcreteAIManager.managers['chains_manager'].get_document_retrieval_chain_with_history(
            llm,
            retrieval_qa_chat_prompt,
            rephrase_prompt,
            vectorstore
        )

        return chain

    @staticmethod
    def retrieve_from_txt_in_cloud(query, chat_history=None):
        """Retrieve information from ingested txt content in vector store."""
        if chat_history is None: chat_history = []

        # 0 - 5
        chain = ConcreteAIManager.get_retrieval_chain()

        # 6
        responses = {}
        for key, prompt in query.items():
            query = prompt
            print(f'before request {key}: {query}')
            response = chain.invoke(input={"input": query, "chat_history": chat_history})

            # Store response for later assertions
            responses[key] = response['answer']

            # ------------------------------------------------------------------------------
            chat_history.append(('human', query))
            chat_history.append(('ai', response['answer']))

            print(f'Response for {key}: {response["answer"][:100]}...')
            print('-' * 10)

        return {
            "responses": responses,
            "chat_history": chat_history
        }

    @staticmethod
    async def cleanup_data():
        """Cleanup Pinecone indexes"""
        all_indexes_in_pinecone = [index_name, index2_name]
        try:
            if not pinecone_api_key or not all_indexes_in_pinecone:
                return {"warning": "Missing Pinecone API key or index names"}

            pc = Pinecone(api_key=pinecone_api_key)
            existing_indexes = [idx.name for idx in pc.list_indexes()]

            cleanup_results = {}
            for idx in all_indexes_in_pinecone:
                if idx in existing_indexes:
                    index = pc.Index(idx)
                    stats = index.describe_index_stats()
                    total_vectors = stats.get('total_vector_count', 0)

                    if total_vectors > 0:
                        index.delete(delete_all=True)
                        cleanup_results[idx] = f"Cleaned up {total_vectors} vectors"
                    else:
                        cleanup_results[idx] = "No vectors to clean up"
                else:
                    cleanup_results[idx] = "Index does not exist"

            return {"success": True, "results": cleanup_results}

        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    async def run_tests(request: dict = None):
        """Run tests programmatically"""
        try:
            test_pattern = request.get("test_pattern") if request else None

            cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]
            if test_pattern:
                cmd.extend(["-k", test_pattern])
            else:
                cmd.append("tests/")

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
