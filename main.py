# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend


# backend = ExampleBackend()
# frontend = StreamlitFrontend(backend)  # runs in a loop

# ---------------------------------------------------------------------------
# LangGraph example
# ---------------------------------------------------------------------------
from langchain_core.prompts import MessagesPlaceholder, SystemMessagePromptTemplate, ChatPromptTemplate
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from support.callback_handler import CustomCallbackHandler
from tests.conftest import get_managers


managers = get_managers()


# 1
tools = [TavilySearch(max_results=1), managers['tools_manager'].triple]

# 3
reflection_prompt_messages = [
    (
        "system",
        "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet. "
        "Always provide detailed recommendations, including requests for length, virality, style, etc.",
    ),
    MessagesPlaceholder(variable_name="messages"),
]
generation_prompt_messages = [
    (
        "system",
        "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
        " Generate the best twitter post possible for the user's request."
        " If the user provides critique, respond with a revised version of your previous attempts.",
    ),
    MessagesPlaceholder(variable_name="messages"),
]

reflection_prompt = managers['prompt_manager'].create_template_from_messages(reflection_prompt_messages)
generation_prompt = managers['prompt_manager'].create_template_from_messages(generation_prompt_messages)


# 4
llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

# 5
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
























# 6
# response = graph.invoke({"messages": query})
