from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from support.callback_handler import CustomCallbackHandler


def create_a_reasoning_and_tool_graph(tools, llm) -> CompiledStateGraph:
    """Create a LangGraph with reasoning and tool nodes."""
    system_message = """You are a helpful assistant that can use tools to answer questions."""

    # ----------------------------------------------------------------------------------
    # State definition
    # ----------------------------------------------------------------------------------
    class MyState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    agent_reason = "agent_reason"
    act = "act"
    last = -1

    # ----------------------------------------------------------------------------------
    # Reasoning node
    # ----------------------------------------------------------------------------------
    def run_agent_reasoning(state: MessagesState) -> MessagesState:
        """
        Run the agent reasoning node.
        :param state: The current state of messages.
        :return: The updated state after reasoning.
        """
        response = llm.invoke(
            [{"role": "system", "content": system_message}, *state['messages']]
        )
        return {'messages': [response]}

    # ----------------------------------------------------------------------------------
    # Tool node
    # ----------------------------------------------------------------------------------
    tool_node = ToolNode(tools)

    # ----------------------------------------------------------------------------------
    # Conditional logic for graph edges
    # ----------------------------------------------------------------------------------
    def should_continue(state: MyState) -> str:
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][last]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return act
        return END

    # ----------------------------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------------------------
    flow = StateGraph(MyState)

    flow.add_node(agent_reason, run_agent_reasoning)
    flow.add_node(act, tool_node)
    flow.set_entry_point(agent_reason)  # the first node to be executed

    flow.add_conditional_edges(
        agent_reason,
        should_continue,
        {END: END, act: act}
    )
    flow.add_edge(act, agent_reason)

    compiled_graph = flow.compile()
    compiled_graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

    return compiled_graph


def test_13_run_graph_with_reasoning_node_and_tool_node(base_dir, managers):
    """Test LangGraph flow with a reasoning node and a tool node."""
    # ----------------------------------------------------------------------------------
    # Arrange & Act
    # ----------------------------------------------------------------------------------
    # 1
    tools = [TavilySearch(max_results=1), managers['tools_manager'].triple]

    # 3
    query = [HumanMessage(content="What is the weather in Tokyo? After fetching the weather, triple the temperature in Celsius.")]

    # 4
    llm = (managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])
           .bind_tools(tools))      # enabled function calling: LLM decides when to call a tool

    # 5
    graph = create_a_reasoning_and_tool_graph(tools, llm)

    # 6
    response = graph.invoke({"messages": query})

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    print(response['messages'][-1].content)

    for msg in response["messages"]:
        if msg.type == "tool":
            print(f"Tool used: {msg.name} with output: {msg.content}")

        # 1️⃣ Ensure there is at least one message in the response
        assert "messages" in response, "Response should have a 'messages' key"
        assert len(response["messages"]) > 0, "Response messages should not be empty"

        # 2️⃣ Ensure the last message is from the AI
        last_msg = response['messages'][-1]
        assert last_msg.type == "ai", "The last message should be an AI message"

        # 3️⃣ Ensure at least one tool was called
        tool_msgs = [msg for msg in response["messages"] if msg.type == "tool"]
        assert len(tool_msgs) > 0, "At least one tool should be called"

        # 4️⃣ Ensure the 'triple' tool was used
        triple_msgs = [msg for msg in tool_msgs if msg.name == "triple"]
        assert len(triple_msgs) > 0, "'triple' tool should be called"

        # 5️⃣ Check that the output of 'triple' is numeric
        triple_output = triple_msgs[0].content
        assert isinstance(triple_output, (int, float, str)), "Triple tool output should be numeric"
        try:
            val = float(triple_output)
        except ValueError:
            assert False, f"Triple tool output is not a valid number: {triple_output}"
