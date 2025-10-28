import os
import datetime
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from models.schemas import AnswerQuestion, ReviseAnswer
from support.callback_handler import CustomCallbackHandler


load_dotenv()


tavily_api_key = os.getenv('TAVILY_API_KEY')


def create_generate_and_revise_and_tool_graph(first_responder_chain, execute_tools, reviser_chain) -> CompiledStateGraph:
    """Create a graph that generates an initial response, executes tools, and then revises the response based on tool outputs."""

    # ----------------------------------------------------------------------------------
    # State definition
    # ----------------------------------------------------------------------------------
    class MyState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    draft_key = "draft"
    execute_tools_key = "execute_tools"
    revise_key = "revise"
    max_iterations = 2

    # ----------------------------------------------------------------------------------
    # Conditional logic for graph edges
    # ----------------------------------------------------------------------------------
    def should_continue(state: MyState) -> str:
        count_tool_visits = sum(isinstance(item, ToolMessage) for item in state['messages'])
        num_iterations = count_tool_visits
        if num_iterations >= max_iterations:
            return END

        return execute_tools_key

    # ----------------------------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------------------------
    flow = StateGraph(MyState)

    flow.add_node(draft_key, first_responder_chain)
    flow.add_node(execute_tools_key, execute_tools)
    flow.add_node(revise_key, reviser_chain)
    flow.set_entry_point(draft_key)

    flow.add_edge(draft_key, execute_tools_key)
    flow.add_edge(execute_tools_key, revise_key)
    flow.add_conditional_edges(
        revise_key,
        should_continue,
        {END: END, execute_tools_key: execute_tools_key}
    )

    compiled_graph = flow.compile()
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

    return compiled_graph


def test_15_run_graph_with_generate_revise_and_tool_nodes(base_dir, managers):
    """Test running a graph that generates an initial response, executes tools, and then revises the response based on tool outputs."""
    # ----------------------------------------------------------------------------------
    # Arrange & Act
    # ----------------------------------------------------------------------------------
    # 1
    tavily_tool = TavilySearch(api_key=tavily_api_key, max_results=5)

    def run_queries(search_queries: list[str], **kwargs):
        """Run the generated queries"""
        return tavily_tool.batch([{"query": q} for q in search_queries])  # batch runs them in parallel

    execute_tools = ToolNode(
        [
            StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
            StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
        ]
    )

    # 3
    system_message_generate = """You are expert researcher.
    Current time: {time}

    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recomment search queries to research information and improve your answer."""

    reviser_instruction = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
            - You MUST include numerical citations in your revised answer to ensure it can be verified.
            - Add a "References" section to the bottom of your answer (which does not count towards the word limit. In the form of:
                - [1] https://example.com
                - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

    generation_prompt_messages = [
        (
            "system", system_message_generate,
        ),
        MessagesPlaceholder(variable_name="messages"),  # Placeholder for conversation history
    ]

    generation_prompt = managers['prompt_manager'].create_template_from_messages(generation_prompt_messages)
    generation_prompt_prefilled = managers['prompt_manager'].prefill_existing_template(
        generation_prompt,
        time=lambda: datetime.datetime.now().isoformat()
    )

    first_responder_prompt_template = managers['prompt_manager'].prefill_existing_template(
        generation_prompt_prefilled,
        first_instruction="Provide a detailed ~250 word answer.",
    )

    reviser_chain_prompt_template = managers['prompt_manager'].prefill_existing_template(
        generation_prompt_prefilled,
        first_instruction=reviser_instruction,
    )

    query = HumanMessage(content="""Write about AI-Powered SOC / autonomous soc problem domain,
    list startups that do that and raised capital.""")

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

    # 5
    first_responder_chain = (
            RunnableLambda(lambda state: {"messages": state["messages"]})
            | first_responder_prompt_template
            | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
            | RunnableLambda(lambda x: {"messages": [x]})
    )

    reviser_chain = (
            RunnableLambda(lambda state: {"messages": state["messages"]})
            | reviser_chain_prompt_template
            | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
            | RunnableLambda(lambda x: {"messages": [x]})
    )

    graph = create_generate_and_revise_and_tool_graph(
        first_responder_chain, execute_tools, reviser_chain
    )

    # 6
    response = graph.invoke({"messages": [query]})

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL REVISED ANSWER:")
    print("=" * 60)

    # Get the last message
    last_message = response['messages'][-1]

    # Check if it has tool calls
    if last_message.tool_calls:
        # Assuming the first tool call is ReviseAnswer
        tool_call = last_message.tool_calls[0]
        if tool_call['name'] == 'ReviseAnswer':
            # tool_call['args'] is already a dictionary, no need to parse JSON
            args = tool_call['args']
            print(args['answer'])
            print("\nReferences:")
            for i, ref in enumerate(args.get('references', []), 1):
                print(f"[{i}] {ref}")
    else:
        print(last_message.content)

    # Basic structure assertions
    assert 'messages' in response, "Response should contain 'messages' key"
    assert isinstance(response['messages'], list), "Messages should be a list"
    assert len(response['messages']) > 1, "Should have multiple messages in the conversation flow"

    # Check that we have a mix of message types throughout the conversation
    message_types = [type(msg).__name__ for msg in response['messages']]
    assert 'HumanMessage' in message_types, "Should contain HumanMessage"
    assert any('AIMessage' in msg_type for msg_type in message_types), "Should contain AIMessage(s)"

    # Verify the graph execution completed successfully
    assert last_message is not None, "Should have a final message"

    # Check if it has tool calls (for ReviseAnswer) or is a final answer
    if last_message.tool_calls:
        # Tool call path assertions
        assert len(last_message.tool_calls) > 0, "Should have at least one tool call"

        tool_call = last_message.tool_calls[0]
        assert tool_call['name'] in ['ReviseAnswer', 'AnswerQuestion'], f"Unexpected tool call: {tool_call['name']}"

        # Verify tool call arguments structure
        args = tool_call['args']
        assert 'answer' in args, "Tool call should contain 'answer' field"
        assert isinstance(args['answer'], str), "Answer should be a string"
        assert len(args['answer'].strip()) > 0, "Answer should not be empty"

        # Check for references if present
        if 'references' in args:
            assert isinstance(args['references'], list), "References should be a list"
            # References should be URLs if present
            for ref in args.get('references', []):
                assert isinstance(ref, str), "Each reference should be a string"
                assert ref.startswith('http'), f"Reference should be a URL: {ref}"

        print(args['answer'])
        print("\nReferences:")
        for i, ref in enumerate(args.get('references', []), 1):
            print(f"[{i}] {ref}")

    else:
        # Direct content path assertions
        assert hasattr(last_message, 'content'), "Last message should have content"
        assert isinstance(last_message.content, str), "Content should be a string"
        assert len(last_message.content.strip()) > 0, "Content should not be empty"
        print(last_message.content)

    # Verify the conversation flow had multiple iterations (tool usage)
    tool_messages = [msg for msg in response['messages'] if isinstance(msg, ToolMessage)]
    ai_messages = [msg for msg in response['messages'] if hasattr(msg, 'tool_calls') and msg.tool_calls]
    human_messages = [msg for msg in response['messages'] if isinstance(msg, HumanMessage)]

    assert len(ai_messages) >= 1, "Should have at least one AI message with tool calls"
    assert len(tool_messages) >= 1, "Should have tool executions in the flow"

    # Verify the graph didn't exceed max iterations
    tool_message_count = sum(isinstance(msg, ToolMessage) for msg in response['messages'])
    assert tool_message_count <= 2, f"Should not exceed max iterations of 2, got {tool_message_count}"

    # Verify the final output contains expected content based on the query
    final_content = last_message.content if not last_message.tool_calls else last_message.tool_calls[0]['args'][
        'answer']
    assert final_content, "Final content should not be empty or None"

    # Check for key topics from the query in the final answer
    query_keywords = ['AI-Powered SOC', 'autonomous soc', 'startups', 'raised capital']
    found_keywords = [keyword for keyword in query_keywords if keyword.lower() in final_content.lower()]
    assert len(found_keywords) >= 2, f"Final answer should address key query topics. Found: {found_keywords}"

    # Verify answer length constraints (approximately 250 words)
    word_count = len(final_content.split())
    assert 100 <= word_count <= 400, f"Answer should be reasonably sized (100-400 words), got {word_count}"

    # Check for citation markers if references are present
    if 'references' in last_message.tool_calls[0]['args'] if last_message.tool_calls else False:
        assert '[' in final_content and ']' in final_content, "Should contain citation markers"
