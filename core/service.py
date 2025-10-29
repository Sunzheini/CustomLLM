"""
AI Service
------------------
Standalone service that executes tasks using LLM calls.
Uses RedisManager for consistent connection management.
"""
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from support.callback_handler import CustomCallbackHandler

load_dotenv()

from shared_lib.contracts.job_schemas import WorkflowGraphState
from shared_lib.needs.INeedRedisManager import INeedRedisManagerInterface
from shared_lib.needs.ResolveNeedsManager import ResolveNeedsManager
from shared_lib.redis_management.redis_manager import RedisManager
from shared_lib.custom_middleware.error_middleware import ErrorMiddleware
from shared_lib.custom_middleware.logging_middleware import EnhancedLoggingMiddleware
from shared_lib.logging_management.logging_manager import LoggingManager

from tests.conftest import split_document, get_managers

# Configuration
AI_QUEUE = os.getenv("AI_QUEUE", "ai_queue")
AI_CALLBACK_QUEUE = os.getenv("AI_CALLBACK_QUEUE", "ai_callback_queue")

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX_NAME')

# Upload/raw storage location constant (configurable)
UPLOAD_DIR = Path(os.getenv("PROCESSED_DIR", "storage/processed")).resolve()

# ai processing constraints
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 500000))  # 10k characters default

logger = LoggingManager.setup_logging(
    service_name="ai-service",
    log_file_path="logs/ai_service.log",
    log_level=logging.INFO,
)


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager to start/stop Redis listener."""
    logger.info("Starting AI Service...")

    # Create RedisManager
    redis_manager = RedisManager()

    # Create ai service and inject RedisManager
    ai_service = AIService()
    ResolveNeedsManager.resolve_needs(ai_service)

    # Store in app.state
    app.state.extract_text_service = ai_service
    app.state.redis_manager = redis_manager

    print("[AIService] Starting Redis listener...")
    logger.info("Starting Redis listener...")
    task = asyncio.create_task(redis_listener(ai_service))
    yield
    print("[AIService] Shutting down Redis listener.")
    logger.info("Shutting down Redis listener.")
    task.cancel()
    await app.state.redis_manager.close()


app = FastAPI(title="AI Service", lifespan=lifespan)
app.add_middleware(ErrorMiddleware)
app.add_middleware(EnhancedLoggingMiddleware, service_name="ai-service")


class AIService(INeedRedisManagerInterface):
    """Handles ai processing tasks using shared RedisManager."""

    def __init__(self):
        self.logger = logging.getLogger("ai-service")
        self.MAX_TEXT_LENGTH = MAX_TEXT_LENGTH

        # Instance-level configuration
        self.base_dir = Path(__file__).resolve().parent.parent
        self.managers = get_managers()

    async def process_ai_task(self, task_data: dict) -> dict:
        """Process ai task using shared Redis connection."""
        try:
            state = WorkflowGraphState(**task_data)
            result_state = await self._process_ai_worker(state)
            return dict(result_state)
        except Exception as e:
            self.logger.error(f"AI processing failed: {e}")
            return {
                "job_id": task_data.get("job_id"),
                "status": "failed",
                "step": "ai_processing_failed",
                "metadata": {"errors": [str(e)]},
                "updated_at": self._current_timestamp(),
            }

    # region AI Processing Methods
    def _split_txt_into_chunks(self, state: WorkflowGraphState) -> list:
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

    def _ingest_txt_into_cloud_vector_store(self, texts: list) -> None:
        """
        Test ingesting txt content into a vector store and querying it.
        This is an integration test that requires OpenAI API access.
        """
        texts = texts[:3]

        # 0
        embeddings = self.managers['embeddings_manager'].open_ai_embeddings()

        # 2
        pinecone_index_name = index_name
        vectorstore = (self.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'create',
            texts, embeddings,
            index_name=pinecone_index_name, pinecone_api_key=pinecone_api_key
        ))

    def _retrieve_from_txt_in_cloud(self):
        embeddings = self.managers['embeddings_manager'].open_ai_embeddings()

        # 2
        pinecone_index_name = index_name
        vectorstore = (self.managers['vector_store_manager']
        .get_vector_store(
            'pinecone', 'load',
            index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
        ))

        # 3
        query = "What microcontrollers are mentioned?"
        retrieval_qa_chat_prompt = self.managers['prompt_manager'].get_prompt_template("langchain-ai/retrieval-qa-chat")

        # 4
        llm = self.managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

        # 5
        chain = self.managers['chains_manager'].get_document_retrieval_chain(llm, retrieval_qa_chat_prompt, vectorstore)

        # 6
        response = chain.invoke(input={"input": query})
        return f"\nAnswer: {response['answer']}"

    # endregion

    async def _process_ai_worker(self, state: WorkflowGraphState) -> WorkflowGraphState:
        """
        Processing with AI.
        Updates the job state with the results of the ai processing.
        Args:
            state (WorkflowGraphState): The job state dictionary containing file metadata and path.
        Returns:
            WorkflowGraphState: Updated job state after ai processing.

        Look like: {
        'job_id': '51be2080-e3e0-4e2b-962d-8d5f2e81c127',
        'file_path': 'D:\\Study\\Projects\\Github\\AegisAI\\shared-storage\\raw\\51be2080-e3e0-4e2b-962d-8d5f2e81c127_P8052AH.pdf',
        'content_type': 'application/pdf',
        'checksum_sha256': '4ee5bec8416491b15f6ff615059e22eb30d0b5662167ce0e972bb84aa1648016',
        'submitted_by': 'Bob',
        'status': 'success',
        'created_at': '2025-10-28T09:40:24.192919+00:00',
        'updated_at': '2025-10-28T09:40:29.095009+00:00',
        'step': 'ai_processing_done',
        'branch': 'pdf_branch',
        'metadata': {
            'validation': 'passed',
            'file_size': 780647,
            'file_extension': '.pdf',
            'created_timestamp':
            '2025-10-28T09:40:24.179919+00:00',
            'modified_timestamp': '2025-10-28T09:40:24.184920+00:00',
            'magic_number_verified': True,
            'page_count': 21,
            'is_encrypted': False,
            'document_info': {
                'ModDate': "D:20060423164721+09'00'",
                'CreationDate': 'D:19970617164339Z',
                'Title': 'DATASHEET SEARCH SITE | WWW.ALLDATASHEET.COM',
                'Creator': 'Acrobat Capture 1.0',
                'Author': 'Provided By ALLDATASHEET.COM(FREE DATASHEET DOWNLOAD SITE)',
                'Keywords': 'PDF, DATASHEET, PDF DATASHEET, IC, CHIP, SEMICONDUCTOR, TRANSISTOR, ELECTRONIC COMPONENT, ISO COMPONENT, ALLDATASHEET, DATABOOK, CATALOG, ARCHIVE',
                'Subject': 'DATASHEET SEARCH, DATABOOK, COMPONENT, FREE DOWNLOAD SITE',
                'Producer': 'Acrobat PDFWriter 2.01 for Windows'},
            'extracting_metadata': 'passed',
            'text_extraction': {
                'success': True,
                'extracted_character_count': 33608,
                'total_pages': 21,
                'pages_with_text': 21,
                'text_file_path': 'D:\\Study\\Projects\\Github\\AegisAI\\shared-storage\\processed\\51be2080-e3e0-4e2b-962d-8d5f2e81c127_extracted_text.txt',
                'file_stats': {
                    'saved_at': '2025-10-28T09:40:27.844876+00:00',
                    'file_size_bytes': 35496,
                    'character_count': 33608,
                    'file_path': 'D:\\Study\\Projects\\Github\\AegisAI\\shared-storage\\processed\\51be2080-e3e0-4e2b-962d-8d5f2e81c127_extracted_text.txt'},
                'content_analysis': {
                    'word_count': 4334,
                    'paragraph_count': 21,
                    'avg_words_per_page': 0,
                    'language_indicators': {},
                    'content_categories': ['technical_document', 'datasheet']},
                'extraction_time': '2025-10-28T09:40:27.844876+00:00',
                'text_preview': '--- Page 1 ---\nnmu @\nMCS@51\n8-BIT CONTROL-ORIENTED MICROCONTROLLERS\nCommercial/Express\n8031AH18051AH18051AHP\n8032N+18052N-I\n8751W8751H-8\n8751BW8752BI-I\nn HighPerformance HMOSProcess n BooleanProcessor\nn Internal Timers/Event Counters n Bit-Addressable RAM\nn 2-Level interrupt Priority Structure n ProgrammableFullDuplexSerial\nChannel\nn 32 1/0 Lines(Four 8-Bit Ports)\nn 111Instructions(64 Single-Cycle)\nn 64K External ProgramMemory Space\n64K External Data Memory Space\nn Security Feature Protects EPRO...'},
                'extract_text': 'passed',
                'ai_processing': {
                    'document_summary': {
                        'summary': 'This is a long detailed_document containing approximately 4334 words. Key topics include: 30pf+10, cycle, abyteofthe.',
                        'word_count': 4334,
                        'sentence_count': 430,
                        'estimated_reading_time_minutes': 21,
                        'key_topics': ['30pf+10', 'cycle', 'abyteofthe', 'frominternalmemory,eais', 'units'],
                        'content_type': 'detailed_document', 'readability_score': 37},
                    'sentiment_analysis': {
                        'sentiment': 'positive',
                        'sentiment_confidence': 15,
                        'tone': 'casual',
                        'positive_indicators': 1,
                        'negative_indicators': 0},
                    'entity_extraction': {
                        'people': [],
                        'organizations': [],
                        'locations': [],
                        'dates': [],
                        'topics': ['technology', 'software', 'systems']},
                    'ai_insights': {
                        'insights': ['The document has a generally positive tone', 'This is a comprehensive document requiring detailed review', 'Content focuses on technical subjects'],
                        'overall_complexity': 'high',
                        'recommended_actions': ['Review key topics', 'Consider sentiment in response']},
                    'processing_timestamp': '2025-10-28T09:40:29.095009+00:00',
                    'model_used': 'dummy_ai_v1.0'},
                'ai_processing_status': 'completed'}}
        """
        print(f"[Worker:ai_processing_file] Job {state['job_id']} ai processing...")
        await asyncio.sleep(0.5)
        errors = []

        # -------------------------------------------------------------------------------
        # The real AI processing!
        # -------------------------------------------------------------------------------
        ai_results = {}

        try:
            texts = self._split_txt_into_chunks(state)

            if not texts:
                errors.append("No text chunks available for AI processing")
            else:
                self._ingest_txt_into_cloud_vector_store(texts)
                summary_response = self._retrieve_from_txt_in_cloud()

                # Compile all AI results
                # Keep the same structure as before but with real AI content
                ai_results.update({
                    "document_summary": {
                        "summary": summary_response.replace('\nAnswer: ', '').strip(),
                        "word_count": len(summary_response.split()),
                        "content_type": "ai_analysis",
                        "readability_score": 85  # You can calculate this
                    },
                    "sentiment_analysis": {
                        "sentiment": "neutral",  # You can analyze this from the response
                        "sentiment_confidence": 75
                    },
                    "entity_extraction": {
                        "topics": ["technical"]  # Extract from your AI response
                    },
                    "ai_insights": {
                        "insights": ["AI analysis completed via vector search"],
                        "overall_complexity": "medium"
                    },
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_used": "gpt-4.1-mini"
                })

        except Exception as e:
            errors.append(f"AI processing failed: {str(e)}")

        # -------------------------------------------------------------------------------
        if "metadata" not in state or state["metadata"] is None:
            state["metadata"] = {}

        if errors:
            state["status"] = "failed"
            state["step"] = "ai_processing_failed"
            state["metadata"]["errors"] = errors

            # Include any partial AI results if available
            if ai_results:
                state["metadata"]["ai_processing"] = {
                    "success": False,
                    "errors": errors,
                    "partial_results": ai_results
                }

        else:
            state["status"] = "success"
            state["step"] = "ai_processing_done"
            state["metadata"]["ai_processing"] = ai_results  # Add AI results to existing metadata
            state["metadata"]["ai_processing_status"] = "completed"  # Add success flag

        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        print(
            f"[Worker:ai_processing] Job {state['job_id']} AI processing done. State: {state}"
        )
        return state

    @staticmethod
    def _current_timestamp():
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai_processing"}


# ----------------------------------------------------------------------------------------------
# Redis listener to subscribe to validation tasks
# ----------------------------------------------------------------------------------------------
async def redis_listener(ai_service: AIService):
    """Redis listener using shared RedisManager."""
    redis_client = await ai_service.redis_manager.get_redis_client()
    pubsub = redis_client.pubsub()

    try:
        await pubsub.subscribe(AI_QUEUE)
        print(f"[AIService] Listening on '{AI_QUEUE}'...")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    task = json.loads(message["data"])
                    job_id = task.get("job_id", "unknown")
                    print(f"[AIService] Processing job: {job_id}")

                    result = await ai_service.process_ai_task(task)

                    # Use shared Redis connection to publish result
                    await redis_client.publish(
                        AI_CALLBACK_QUEUE,
                        json.dumps({"job_id": job_id, "result": result}),
                    )
                    print(f"[AIService] Published result for: {job_id}")

                except Exception as e:
                    print(f"[AIService] Error: {e}")

    except asyncio.CancelledError:
        print("[AIService] Listener cancelled")
    finally:
        await pubsub.unsubscribe(AI_QUEUE)
        await pubsub.close()
