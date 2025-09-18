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
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from langgraph_example.nodes import run_agent_reasoning, tool_node


class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MyState):
    """Determine if we should continue to tools or end."""
    messages = state['messages']
    last_message = messages[-1]

    # If there are tool calls, continue to ACT
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return ACT
    # Otherwise, end
    return "__end__"


# Create the graph
flow = StateGraph(MyState)

# Add nodes
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

# Add edges
flow.add_edge("__start__", AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue)
flow.add_edge(ACT, AGENT_REASON)

# Compile the graph
app = flow.compile()


# Example usage
if __name__ == "__main__":
    # Test the graph
    result = app.invoke({
        # "messages": [HumanMessage(content="What is 5 tripled?")]
        "messages": [HumanMessage(content="Who is the current president of the United States?")]
    })
    print(result)

    for msg in result["messages"]:
        if msg.type == "tool":
            print(f"Tool used: {msg.name} with output: {msg.content}")
