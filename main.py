# ---------------------------------------------------------------------------
# Streamlit example (use the streamlit_runner in the configurations to run)
# ---------------------------------------------------------------------------
from langchain_core.runnables import Runnable

from models.schemas import AnswerQuestion
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

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser

from support.callback_handler import CustomCallbackHandler
from tests.conftest import get_managers


managers = get_managers()


def create__graph(chains) -> CompiledStateGraph:
    # generation_chain, reflection_chain = chains

    # ----------------------------------------------------------------------------------
    # State definition
    # ----------------------------------------------------------------------------------
    # class MyState(TypedDict):
    #     messages: Annotated[list[BaseMessage], add_messages]
    #
    # reflect = "reflect"
    # generate = "generate"
    # max_message_length = 6

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
    # def should_continue(state: MyState) -> str:
    #     if len(state["messages"]) > max_message_length:
    #         return END
    #
    #     return reflect

    # ----------------------------------------------------------------------------------
    # Build graph
    # ----------------------------------------------------------------------------------
    # flow = StateGraph(state_schema=MyState)
    #
    # flow.add_node(generate, run_agent_generation)
    # flow.add_node(reflect, run_agent_reflection)
    # flow.set_entry_point(generate)
    #
    # flow.add_conditional_edges(
    #     generate,
    #     should_continue,
    #     {END: END, reflect: reflect}
    # )
    # flow.add_edge(reflect, generate)
    #
    # compiled_graph = flow.compile()
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path='flow.png')
    #
    # return compiled_graph
    pass


def run_it():
    # 3
    system_message_actor = """You are expert researcher.
    Current time: {time}
    
    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recomment search queries to research information and improve your answer."""

    # system_message_reviser = """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
    # Always provide detailed recommendations, including requests for length, virality, style, etc.
    # """

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
    first_responder_chain = first_responder_prompt_template | llm.bind_tools(
        tools=[AnswerQuestion],
        tool_choice="AnswerQuestion",
    )

    # response = graph.invoke({"messages": [query]})
    # response = first_responder_chain.invoke({"messages": [query]})

    new_chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | pydantic_parser
    )

    response = new_chain.invoke({"messages": [query]})

    # print(response['messages'])
    # print(response['messages'][-1].content)
    print(response[0].answer)


run_it()
