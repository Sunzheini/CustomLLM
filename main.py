# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from langchain_core.runnables import Runnable

from streamlit_example.example_backend import ExampleBackend
from streamlit_example.example_streamlit_frontend import StreamlitFrontend


# backend = ExampleBackend()
# frontend = StreamlitFrontend(backend)  # runs in a loop


# ---------------------------------------------------------------------------
# LangGraph example: Reflexion architecture
# ---------------------------------------------------------------------------
from langchain_core.prompts import MessagesPlaceholder
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from support.callback_handler import CustomCallbackHandler
from tests.conftest import get_managers


managers = get_managers()


def create__graph(chains) -> CompiledStateGraph:
    generation_chain, reflection_chain = chains

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
    compiled_graph.get_graph().draw_mermaid_png(output_file_path='flow.png')

    return compiled_graph


def run_it():
    system_message_generate = """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
    Generate the best twitter post possible for the user's request.
    If the user provides critique, respond with a revised version of your previous attempts.
    """
    system_message_reflect = """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
    Always provide detailed recommendations, including requests for length, virality, style, etc.
    """

    # 1
    tools = [TavilySearch(max_results=1), managers['tools_manager'].triple]

    # 3
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

    graph = create__graph([generation_chain, reflection_chain])

    # 6
    response = graph.invoke({"messages": [query]})

    print(response['messages'])
    print(response['messages'][-1].content)


run_it()
