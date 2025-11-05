"""
AI Service
------------------
Standalone service that executes tasks using LLM calls.
Uses RedisManager for consistent connection management.

Note: This service is different from the base ones@api-gateway-service project with
the real AI processing and 3 additional endpoints for generating responses, cleaning
up data and running tests.
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
from starlette.responses import JSONResponse

load_dotenv()

from shared_lib.contracts.job_schemas import WorkflowGraphState
from shared_lib.needs.INeedRedisManager import INeedRedisManagerInterface
from shared_lib.needs.ResolveNeedsManager import ResolveNeedsManager
from shared_lib.redis_management.redis_manager import RedisManager
from shared_lib.custom_middleware.error_middleware import ErrorMiddleware
from shared_lib.custom_middleware.logging_middleware import EnhancedLoggingMiddleware
from shared_lib.logging_management.logging_manager import LoggingManager

# get the concrete manager
from core.concrete_ai_manager import ConcreteAIManager


# Configuration
AI_QUEUE = os.getenv("AI_QUEUE", "ai_queue")
AI_CALLBACK_QUEUE = os.getenv("AI_CALLBACK_QUEUE", "ai_callback_queue")

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

        # query = {
        #     'query1': "What microcontrollers are mentioned?",
        #     'query2': "What did I just ask you?",
        #     'query3': "What is the first microcontroller listed in your first reply to me?",
        # }

        # Improved queries for better document analysis
        query = {
            'summary': "Provide a comprehensive executive summary of this document, highlighting main topics and key information",
            'key_points': "Extract the most important technical specifications, features, or key findings mentioned in the document",
            'analysis': "Identify the document's primary purpose, target audience, structure, and any critical recommendations"
        }

        chat_history = []  # list of (user, bot) tuples

        try:
            texts = ConcreteAIManager.split_txt_into_chunks(state)

            if not texts:
                errors.append("No text chunks available for AI processing")
            else:
                ConcreteAIManager.ingest_txt_into_cloud_vector_store(texts)
                summary_response = ConcreteAIManager.retrieve_from_txt_in_cloud(query, chat_history)

                all_responses = summary_response["responses"]

                # Create a proper document summary from all responses
                combined_summary = f"""
                DOCUMENT OVERVIEW:
                {all_responses['summary']}

                KEY POINTS:
                {all_responses['key_points']}

                DOCUMENT ANALYSIS:
                {all_responses['analysis']}
                """

                actual_word_count = len(combined_summary.split())

                # Compile all AI results
                ai_results.update({
                    "document_summary": {
                        "summary": combined_summary,
                        "executive_summary": all_responses['summary'],
                        "key_points": all_responses['key_points'],
                        "document_analysis": all_responses['analysis'],
                        "word_count": actual_word_count,
                        "content_type": "comprehensive_analysis",
                        "readability_score": min(100, max(50, 100 - (actual_word_count // 10)))
                    },
                    "sentiment_analysis": {
                        "sentiment": "neutral",  # You could analyze this from the content
                        "sentiment_confidence": 75
                    },
                    "entity_extraction": {
                        "topics": ["technical_document"]  # Extract actual topics from content
                    },
                    "ai_insights": {
                        "insights": ["Document analyzed via vector search and multi-perspective queries"],
                        "overall_complexity": "medium",
                        "analysis_depth": "comprehensive"
                    },
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_used": "gpt-4.1-mini",
                    "analysis_queries_used": 3
                })

        except Exception as e:
            errors.append(f"AI processing failed: {str(e)}")
            self.logger.error(f"AI processing error: {str(e)}")

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


# -----------------------------------------------------------------------------------------------
# FastAPI endpoints for health check, cleanup and test execution
# -----------------------------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai_processing"}


@app.post("/generate-response")
async def generate_response(request: dict):
    try:
        query = request.get("user_prompt")
        chat_history = request.get("chat_history", [])

        if not query:
            return {"error": "user_prompt is required"}

        # 0 - 5
        chain = ConcreteAIManager.get_retrieval_chain()

        # 6
        response = chain.invoke(input={"input": query, "chat_history": chat_history})

        result = {
            "query": query,
            "result": response['answer'],
            "source_documents": response.get('context', []),
            "chat_history": chat_history + [('human', query), ('ai', response['answer'])]
        }

        return result

    except Exception as e:
        return {"error": str(e)}


@app.post("/clean")
async def cleanup_data():
    """Cleanup Pinecone indexes (POST endpoint)"""
    try:
        result = await ConcreteAIManager.cleanup_data()

        if result is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Cleanup operation returned no result"}
            )

        if "error" in result:
            return JSONResponse(
                status_code=400,
                content=result
            )

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Cleanup operation failed: {str(e)}"}
        )


@app.post("/run-tests")
async def run_tests(request: dict = None):
    """Run tests programmatically (POST endpoint)"""
    await ConcreteAIManager.run_tests()


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

