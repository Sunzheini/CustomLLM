# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
import os
from pathlib import Path

from dotenv import load_dotenv
from firecrawl.v2.methods.aio.search import search
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.prebuilt import ToolNode

from models.schemas import AnswerQuestion, ReviseAnswer
from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend


# backend = ExampleBackend()
# frontend = StreamlitFrontend(backend)  # runs in a loop


# ---------------------------------------------------------------------------
# LangGraph example: Reflexion architecture
# ---------------------------------------------------------------------------
import datetime

from langchain_core.prompts import MessagesPlaceholder
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langgraph.graph import StateGraph, MessagesState, END, MessageGraph

from support.callback_handler import CustomCallbackHandler
from tests.conftest import get_managers


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')


managers = get_managers()


def create_graph() -> CompiledStateGraph:
    pass


def run_it():
    pass


run_it()
