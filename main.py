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


def create_graph(first_responder_chain, execute_tools, reviser_chain) -> CompiledStateGraph:
    # generation_chain, reflection_chain = chains

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
    # Generation node
    # ----------------------------------------------------------------------------------
    # def run_agent_generation(state: MyState) -> MyState:
    #     response = generation_chain.invoke({"messages": state["messages"]})
    #     return {'messages': [response]}

    # ----------------------------------------------------------------------------------
    # Reflection node
    # ----------------------------------------------------------------------------------
    # def run_agent_reflection(state: MyState) -> MyState:
    #     response = reflection_chain.invoke({"messages": state["messages"]})
    #     return {'messages': [HumanMessage(content=response.content)]}   # cast to HumanMessage to think this is user input

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


def run_it():
    # 1
    tavily_tool = TavilySearch(api_key=tavily_api_key, max_results=5)

    # tools = [TavilySearch(api_key=tavily_api_key, max_results=5)]

    def run_queries(search_queries: list[str], **kwargs):
        """Run the generated queries"""
        return tavily_tool.batch([{"query": q} for q in search_queries])    # batch runs them in parallel

    execute_tools = ToolNode(
        [
            StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
            StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
        ]
    )

    # 3
    system_message_actor = """You are expert researcher.
    Current time: {time}
    
    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recomment search queries to research information and improve your answer."""

    system_message_reviser = """Revise your previous answer using the new information.
        - You should use the previous critique to add important information to your answer.
            - You MUST include numerical citations in your revised answer to ensure it can be verified.
            - Add a "References" section to the bottom of your answer (which does not count towards the word limit. In the form of:
                - [1] https://example.com
                - [2] https://example.com
        - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
    """

    actor_prompt_messages = [
        (
            "system", system_message_actor,
        ),
        MessagesPlaceholder(variable_name="messages"),      # Placeholder for conversation history
    ]
    # reviser_prompt_messages = [
    #     (
    #         "system", system_message_reviser,
    #     ),
    #     MessagesPlaceholder(variable_name="messages"),
    # ]

    actor_prompt = managers['prompt_manager'].create_template_from_messages(actor_prompt_messages)
    actor_prompt_prefilled = managers['prompt_manager'].prefill_existing_template(
        actor_prompt,
        time=lambda: datetime.datetime.now().isoformat()
    )

    # reviser_prompt = managers['prompt_manager'].create_template_from_messages(reviser_prompt_messages)

    query = HumanMessage(content="""Write about AI-Powered SOC / autonomous soc problem domain,
    list startups that do that and raised capital.""")

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

    # 5
    json_parser = JsonOutputToolsParser(return_id=True)  # return the f-call we got from the llm and transform it to a dict
    pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])   # converts to a AnswerQuestion object

    # generation_chain = generation_prompt | llm
    # reflection_chain = reflection_prompt | llm

    # graph = create__graph([generation_chain, reflection_chain])

    # 6
    first_responder_prompt_template = managers['prompt_manager'].prefill_existing_template(
        actor_prompt_prefilled,
        first_instruction="Provide a detailed ~250 word answer.",
    )
    first_responder_chain = (
            RunnableLambda(lambda state: {"messages": state["messages"]})
            | first_responder_prompt_template
            | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
            | RunnableLambda(lambda x: {"messages": [x]})
    )

    reviser_chain_prompt_template = managers['prompt_manager'].prefill_existing_template(
        actor_prompt_prefilled,
        first_instruction=system_message_reviser,
    )

    reviser_chain = (
            RunnableLambda(lambda state: {"messages": state["messages"]})
            | reviser_chain_prompt_template
            | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
            | RunnableLambda(lambda x: {"messages": [x]})
    )

    # response = graph.invoke({"messages": [query]})
    # response = first_responder_chain.invoke({"messages": [query]})

    new_chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | pydantic_parser
    )

    graph = create_graph(
        first_responder_chain, execute_tools, reviser_chain
    )

    response = graph.invoke({"messages": [query]})

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


run_it()
