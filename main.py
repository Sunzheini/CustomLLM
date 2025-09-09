import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from core.command_menu import CommandMenu
from models.schemas import AgentResponse
from prompts.prompt1 import CUSTOM_USER_PROMPT
from prompts.prompt2 import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from support.measure_and_print_time_decorator import measure_and_print_time_decorator


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

open_ai_api_key = os.getenv('OPENAI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')


@measure_and_print_time_decorator
def function_1():
    # -------------------------------------------------------------------------------------------------------
    # prompt templates
    # -------------------------------------------------------------------------------------------------------
    # 1. predefined react agent prompt template
    # react_prompt_template = hub.pull("hwchase17/react")

    # 2. predefined react agent prompt template customized with output instructions
    output_parser = PydanticOutputParser(
        pydantic_object=AgentResponse,  # the model that defines the output structure
    )
    react_prompt_template = PromptTemplate(
        # template=CUSTOM_USER_PROMPT,
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        # input_variables=["question"],  # which variable will be provided at runtime
        input_variables=["question", "agent_scratchpad", "tool_names"],
    ).partial(
        format_instructions=output_parser.get_format_instructions()     # format_instructions is provided in advance
    )

    # -------------------------------------------------------------------------------------------------------
    # LLMs
    # -------------------------------------------------------------------------------------------------------
    # temperature=0 for deterministic, 1 for creative
    llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")        # small model
    # llm = ChatOllama(temperature=0, model="gpt-oss:20b")        # large model, dont run until 32GB RAM
    # llm = ChatOllama(temperature=0, model="gemma3:4b")          # medium, slow


    # -------------------------------------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------------------------------------
    # tools allow llms to access external utilities
    tools = [TavilySearch(api_key=tavily_api_key),]


    # --------------------------------------------------------------------------------------------------------
    # Chains / Agents: RunnableSequence, AgentExecutor: chains of calls
    # --------------------------------------------------------------------------------------------------------
    # 1. Custom chain with prompt template and llm
    # chain = summary_prompt_template | llm   # output of one is input to the next

    # 2. React agent chain with llm, tools and react prompt template
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    chain = agent_executor

    # -------------------------------------------------------------------------------------------------------
    # Request and response
    # AgentExecutor response is always a dict with 'input' and 'output' keys
    # -------------------------------------------------------------------------------------------------------
    response = chain.invoke(input={
        # "info": info
        "question": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        # "question": "provide 3 random integers"
        # "question": "Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion. Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen."
    })

    """
    Final Answer: 
    ```json
    {
      "answer": "Here are 3 random integers: 7, 42, 19.",
      "sources": []
    }
    ```
    """

    # print(response.content)
    print(response)


    # -------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # menu = CommandMenu({
    #     '1': function_1,
    # })
    function_1()
    # menu.run()
