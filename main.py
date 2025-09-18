# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from pprint import pprint

from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend


# backend = ExampleBackend()
# frontend = StreamlitFrontend(backend)  # runs in a loop

# ---------------------------------------------------------------------------
# LangGraph example
# ---------------------------------------------------------------------------
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

from langgraph_example.nodes import run_agent_reasoning, tool_node


class MyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MyState) -> str:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][LAST]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return ACT
    return END


# Create the graph
flow = StateGraph(MyState)

# Add nodes
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

# Add edges
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT
    }
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path='./langgraph_example/flow.png')   # create a PNG of the graph


if __name__ == "__main__":
    result = app.invoke({
        # "messages": [HumanMessage(content="What is 5 tripled?")]
        # "messages": [HumanMessage(content="Who is the current president of the United States?")]
        "messages": [HumanMessage(content="What is the weather in Tokyo? After fetching the weather, triple the temperature in Celsius.")]
    })
    pprint(result)
    print(80 * '-')
    print(result['messages'][-1].content)
    print(80 * '-')

    for msg in result["messages"]:
        if msg.type == "tool":
            print(f"Tool used: {msg.name} with output: {msg.content}")
