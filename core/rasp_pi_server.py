"""
Voice Assistant AI Service
FastAPI service that handles voice queries from Raspberry Pi with LangGraph workflow
"""
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, TypedDict, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import requests
from starlette.responses import JSONResponse

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice-assistant-service")


# Pydantic models for API
class VoiceQueryRequest(BaseModel):
    query: str
    chat_history: List[tuple] = []
    user_id: Optional[str] = None


class VoiceQueryResponse(BaseModel):
    answer: str
    chat_history: List[tuple]
    needs_web_search: bool
    sources_used: List[str]
    processing_time: float


# LangGraph State Definition
class AssistantState(TypedDict):
    query: str
    chat_history: List[tuple]
    needs_web_search: bool
    search_terms: List[str]
    search_results: str
    initial_response: str
    final_response: str
    sources_used: List[str]
    error: Optional[str]


class VoiceAssistantService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=OPENAI_API_KEY
        )
        self.search_tool = DuckDuckGoSearchRun()
        self.workflow = self._create_workflow()
        logger.info("Voice Assistant Service initialized")

    def _create_workflow(self):
        """Create LangGraph workflow for voice query processing"""

        def analyze_query(state: AssistantState):
            """Analyze if query needs web search"""
            query = state["query"].lower()

            # Keywords that indicate need for current information
            search_keywords = [
                'current', 'latest', 'today', 'recent', 'news', 'weather',
                'what\'s happening', 'update', 'now', '2024', '2025'
            ]

            # Question words that often need facts
            question_words = ['who is', 'what is', 'when did', 'where is', 'how to']

            needs_search = (
                    any(keyword in query for keyword in search_keywords) or
                    any(query.startswith(word) for word in question_words) or
                    len(query.split()) > 10  # Complex queries might need search
            )

            # Extract search terms
            search_terms = [query]
            if needs_search:
                # Simple search term extraction - remove question words
                clean_query = query
                for word in question_words:
                    clean_query = clean_query.replace(word, '')
                search_terms = [clean_query.strip()]

            return {
                **state,
                "needs_web_search": needs_search,
                "search_terms": search_terms,
                "sources_used": []
            }

        def perform_search(state: AssistantState):
            """Perform web search if needed"""
            if not state["needs_web_search"]:
                return {**state, "search_results": ""}

            try:
                search_query = " ".join(state["search_terms"])
                logger.info(f"Searching web for: {search_query}")

                results = self.search_tool.run(search_query)

                # Limit results length for context
                if len(results) > 1000:
                    results = results[:1000] + "..."

                return {
                    **state,
                    "search_results": results,
                    "sources_used": state["sources_used"] + ["web_search"]
                }
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {**state, "search_results": f"Search unavailable: {str(e)}"}

        def generate_initial_response(state: AssistantState):
            """Generate initial response using LLM"""
            try:
                # Prepare context
                context_parts = []

                if state["search_results"]:
                    context_parts.append(f"Search Results:\n{state['search_results']}")

                # Add recent chat history for context
                recent_history = state["chat_history"][-4:]  # Last 2 exchanges
                if recent_history:
                    history_text = "\n".join([f"{speaker}: {msg}" for speaker, msg in recent_history])
                    context_parts.append(f"Recent Conversation:\n{history_text}")

                context = "\n\n".join(context_parts) if context_parts else "No additional context"

                # Create prompt
                prompt = f"""
                You are a helpful voice assistant. Answer the user's question clearly and concisely.
                The response will be converted to speech, so keep it natural and conversational.

                User Question: {state['query']}

                Additional Context:
                {context}

                Guidelines:
                - Be direct and clear
                - Keep responses under 100 words for voice delivery
                - Use natural, conversational language
                - If using search results, summarize key points
                - If unsure, say so rather than guessing

                Response:
                """

                # Get LLM response
                response = self.llm.invoke([HumanMessage(content=prompt)])
                initial_response = response.content.strip()

                return {
                    **state,
                    "initial_response": initial_response,
                    "sources_used": state["sources_used"] + ["llm"]
                }

            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                return {
                    **state,
                    "error": f"Response generation failed: {str(e)}",
                    "initial_response": "I apologize, but I'm having trouble generating a response right now."
                }

        def revise_for_voice(state: AssistantState):
            """Revise response to be optimal for voice delivery"""
            if state.get("error"):
                return {**state, "final_response": state["initial_response"]}

            try:
                revision_prompt = f"""
                Revise this response to be perfect for voice delivery:

                Original Response: {state['initial_response']}

                Make it:
                - More conversational and natural sounding
                - Clear and easy to understand when spoken
                - Appropriately paced (not too dense)
                - Friendly but professional

                Revised Response:
                """

                response = self.llm.invoke([HumanMessage(content=revision_prompt)])
                final_response = response.content.strip()

                return {**state, "final_response": final_response}

            except Exception as e:
                logger.error(f"Revision failed: {e}")
                return {**state, "final_response": state["initial_response"]}

        # Build the graph
        workflow = StateGraph(AssistantState)

        # Add nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("perform_search", perform_search)
        workflow.add_node("generate_response", generate_initial_response)
        workflow.add_node("revise_response", revise_for_voice)

        # Define flow
        workflow.set_entry_point("analyze_query")

        # Conditional routing based on search need
        workflow.add_conditional_edges(
            "analyze_query",
            lambda state: "perform_search" if state["needs_web_search"] else "generate_response"
        )

        workflow.add_edge("perform_search", "generate_response")
        workflow.add_edge("generate_response", "revise_response")
        workflow.add_edge("revise_response", END)

        return workflow.compile()

    async def process_voice_query(self, request: VoiceQueryRequest) -> VoiceQueryResponse:
        """Process a voice query through the LangGraph workflow"""
        start_time = datetime.now(timezone.utc)

        try:
            # Prepare initial state
            initial_state = {
                "query": request.query,
                "chat_history": request.chat_history,
                "needs_web_search": False,
                "search_terms": [],
                "search_results": "",
                "initial_response": "",
                "final_response": "",
                "sources_used": [],
                "error": None
            }

            # Execute workflow
            logger.info(f"Processing query: {request.query}")
            result = await self.workflow.ainvoke(initial_state)

            # Update chat history
            new_chat_history = request.chat_history + [
                ('human', request.query),
                ('ai', result["final_response"])
            ]

            # Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return VoiceQueryResponse(
                answer=result["final_response"],
                chat_history=new_chat_history,
                needs_web_search=result["needs_web_search"],
                sources_used=result["sources_used"],
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# FastAPI App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown management"""
    # Initialize service
    app.state.assistant_service = VoiceAssistantService()
    logger.info("Voice Assistant Service started")

    yield

    # Cleanup
    logger.info("Voice Assistant Service stopped")


app = FastAPI(
    title="Voice Assistant AI Service",
    description="AI service for Raspberry Pi voice assistant with LangGraph workflow",
    version="1.0.0",
    lifespan=lifespan
)


# API Routes
@app.post("/query", response_model=VoiceQueryResponse)
async def process_query(request: VoiceQueryRequest):
    """Main endpoint for voice queries"""
    return await app.state.assistant_service.process_voice_query(request)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "voice-assistant",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/clear-context")
async def clear_context(user_id: str):
    """Clear conversation context for a user"""
    # In a real implementation, you'd clear from database
    return {"status": "context_cleared", "user_id": user_id}


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
