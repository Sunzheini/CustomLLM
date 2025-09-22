from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import MessagesPlaceholder

from support.callback_handler import CustomCallbackHandler


def create_a_generate_and_reflect_graph(generation_chain, reflection_chain) -> CompiledStateGraph:
    """Create a LangGraph flow with generate and reflect nodes."""
    # system_message = """You are a helpful assistant that can use tools to answer questions."""

    # ----------------------------------------------------------------------------------
    # State definition
    # ----------------------------------------------------------------------------------
    class MyState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    reflect = "reflect"
    generate = "generate"
    max_message_length = 6

    # ----------------------------------------------------------------------------------
    # Generation node
    # ----------------------------------------------------------------------------------
    def run_agent_generation(state: MyState) -> MyState:
        response = generation_chain.invoke({"messages": state["messages"]})
        return {'messages': [response]}

    # ----------------------------------------------------------------------------------
    # Reflection node
    # ----------------------------------------------------------------------------------
    def run_agent_reflection(state: MyState) -> MyState:
        response = reflection_chain.invoke({"messages": state["messages"]})
        return {'messages': [HumanMessage(content=response.content)]}   # cast to HumanMessage to think this is user input

    # ----------------------------------------------------------------------------------
    # Conditional logic for graph edges
    # ----------------------------------------------------------------------------------
    def should_continue(state: MyState) -> str:
        if len(state["messages"]) > max_message_length:
            return END

        return reflect

    # ----------------------------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------------------------
    flow = StateGraph(state_schema=MyState)

    flow.add_node(generate, run_agent_generation)
    flow.add_node(reflect, run_agent_reflection)
    flow.set_entry_point(generate)

    flow.add_conditional_edges(
        generate,
        should_continue,
        {END: END, reflect: reflect}
    )
    flow.add_edge(reflect, generate)

    compiled_graph = flow.compile()
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

    return compiled_graph


def test_13_run_graph_with_reasoning_node_and_tool_node(base_dir, managers):
    """Test LangGraph flow with a generation node followed by a reflection node."""
    # ----------------------------------------------------------------------------------
    # Arrange & Act
    # ----------------------------------------------------------------------------------
    # 3
    system_message_generate = """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
    Generate the best twitter post possible for the user's request.
    If the user provides critique, respond with a revised version of your previous attempts.
    """
    system_message_reflect = """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
    Always provide detailed recommendations, including requests for length, virality, style, etc.
    """

    generation_prompt_messages = [
        (
            "system", system_message_generate,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
    reflection_prompt_messages = [
        (
            "system", system_message_reflect,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]

    generation_prompt = managers['prompt_manager'].create_template_from_messages(generation_prompt_messages)
    reflection_prompt = managers['prompt_manager'].create_template_from_messages(reflection_prompt_messages)

    query = HumanMessage(content="""Make this tweet better:"
                                        @LangChainAI
                - newly Tool Calling feature is seriously underrated.

                After a long wait, it's here- making the implementation of agents across different models with function calling - super easy.

                Made a video covering their newest blog post

                                    """)

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

    # 5
    generation_chain = generation_prompt | llm
    reflection_chain = reflection_prompt | llm

    graph = create_a_generate_and_reflect_graph(generation_chain, reflection_chain)

    # 6
    response = graph.invoke({"messages": query})

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    print(response['messages'][-1].content)

    # 1️⃣ Ensure there is at least one message in the response
    assert "messages" in response, "Response should have a 'messages' key"
    assert len(response["messages"]) > 0, "Response messages should not be empty"

    # 2️⃣ Ensure the response contains multiple iterations (generate -> reflect -> generate)
    assert len(response["messages"]) >= 3, "Should have at least 3 messages (initial, generation, reflection)"

    # 3️⃣ Ensure we have both generation and reflection messages
    ai_messages = [msg for msg in response["messages"] if msg.type == "ai"]
    human_messages = [msg for msg in response["messages"] if msg.type == "human"]

    assert len(ai_messages) > 0, "Should have at least one AI generation message"
    assert len(human_messages) > 0, "Should have at least one human-style reflection message"

    # 4️⃣ Ensure the last message is from the AI (final generation)
    last_msg = response['messages'][-1]
    assert last_msg.type == "ai", "The last message should be an AI message (final generation)"

    # 5️⃣ Verify content quality - the final tweet should be improved
    final_tweet = last_msg.content
    assert len(final_tweet) > 50, "Final tweet should be substantial in length"
    assert "@LangChainAI" in final_tweet or "LangChain" in final_tweet, "Should mention LangChain"
    assert "tool calling" in final_tweet.lower() or "function calling" in final_tweet.lower(), "Should mention tool/function calling"

    # 6️⃣ Check that the graph went through multiple iterations
    # The pattern should be: Human (query) -> AI (gen1) -> Human (reflect) -> AI (gen2) -> ...
    message_types = [msg.type for msg in response["messages"]]
    assert message_types.count("ai") >= 2, "Should have multiple generation iterations"
    assert message_types.count("human") >= 2, "Should have multiple reflection iterations (including initial query)"

    # 7️⃣ Verify the reflection messages contain critique elements
    reflection_contents = [msg.content for msg in response["messages"] if msg.type == "human" and msg != query]
    if reflection_contents:  # Only check if we have reflection messages
        reflection_text = " ".join(reflection_contents).lower()
        critique_keywords = ["improve", "better", "suggest", "recommend", "critique", "feedback", "length", "style",
                             "viral"]
        has_critique = any(keyword in reflection_text for keyword in critique_keywords)
        assert has_critique, "Reflection messages should contain critique or recommendations"
