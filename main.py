import os
import json
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.chains.question_answering.map_rerank_prompt import output_parser
from langchain_community.document_loaders import TextLoader
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.agents import tool
from langchain_core.tools import render_text_description
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
from prompts.prompt3 import PROMPT3
from support.callback_handler import CustomCallbackHandler
from support.measure_and_print_time_decorator import measure_and_print_time_decorator


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

open_ai_api_key = os.getenv('OPENAI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
huggingface_hub = os.getenv('HUGGINGFACEHUB_API_TOKEN')


@measure_and_print_time_decorator
def function_1():
    # -------------------------------------------------------------------------------------------------------
    # Ingestion (ingest the exampleblog.txt into a vector db
    # -------------------------------------------------------------------------------------------------------
    path_to_file = os.path.join(BASE_DIR, 'context', 'exampleblog.txt')
    loader = TextLoader(path_to_file)
    documents = loader.load()



    # -------------------------------------------------------------------------------------------------------
    # Tools
    # -------------------------------------------------------------------------------------------------------
    # tools allow llms to access external utilities
    # tools = [TavilySearch(api_key=tavily_api_key),]


    # 3. you can define your own tools
    @tool
    def get_text_length(text: str) -> int:
        """Returns the length of the input text."""     # llm uses this description to decide when to use this tool
        print(f"get_text_length enter with {text=}")
        text = text.strip("'\n").strip(
            '"'
        )   # clean up the input text from extra quotes and newlines
        return len(text)


    tools = [get_text_length, ]


    # -------------------------------------------------------------------------------------------------------
    # prompt templates
    # -------------------------------------------------------------------------------------------------------
    # 1. predefined react agent prompt template
    # react_prompt_template = hub.pull("hwchase17/react")

    # 2. predefined react agent prompt template customized with output instructions
    # output_parser = PydanticOutputParser(
    #     pydantic_object=AgentResponse,  # the model that defines the output structure
    # )
    # react_prompt_template = PromptTemplate(
    #     # template=CUSTOM_USER_PROMPT,
    #     template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    #     # input_variables=["question"],  # which variable will be provided at runtime
    #     input_variables=["question", "agent_scratchpad", "tool_names"],
    # ).partial(
    #     format_instructions=output_parser.get_format_instructions()     # format_instructions is provided in advance
    # )

    # 3
    template3 = PromptTemplate.from_template(   # `from_template` Automatically detects variables from the template string
        template=PROMPT3,
    ).partial(
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    # -------------------------------------------------------------------------------------------------------
    # LLMs
    # -------------------------------------------------------------------------------------------------------
    # temperature=0 for deterministic, 1 for creative
    # llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
    # llm = ChatOllama(temperature=0, model="gemma3:270m")        # small model
    # llm = ChatOllama(temperature=0, model="gpt-oss:20b")        # large model, dont run until 32GB RAM
    # llm = ChatOllama(temperature=0, model="gemma3:4b")          # medium, slow

    # 3
    llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini", callbacks=[CustomCallbackHandler()]).bind(
        stop=["\nObservation:", "Observation"]  # # Safety net, not always triggered, stop sequence to end generation when the model outputs anything from the list
    )


    # --------------------------------------------------------------------------------------------------------
    # Chains / Agents: RunnableSequence, AgentExecutor: chains of calls
    # --------------------------------------------------------------------------------------------------------
    # 2. React agent chain with llm, tools and react prompt template
    # agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_template)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    #
    # extract_output = RunnableLambda(lambda x: x['output'])              # extract the 'output' field from the AgentExecutor response
    # parse_output = RunnableLambda(lambda x: output_parser.parse(x))     # parse the JSON string into the Pydantic model
    #
    # # chain = agent_executor
    # chain = agent_executor | extract_output | parse_output   # chain the agent executor with output extraction and parsing

    # 3a
    # chain = template3 | llm     # `|`: output of one is input to the next (Part of the LangCHain Expression Language)
    agent = create_react_agent(llm=llm, tools=tools, prompt=template3)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)  # also return intermediate steps for inspection
    chain = agent_executor


    # -------------------------------------------------------------------------------------------------------
    # Request and response
    # AgentExecutor response is always a dict with 'input' and 'output' keys
    # -------------------------------------------------------------------------------------------------------
    # 2
    # response = chain.invoke(input={
    #     # "info": info
    #     "question": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    #     # "question": "provide 3 random integers"
    #     # "question": "Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion. Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen."
    # })
    # print(response)

    # 3
    response = chain.invoke(
        input={
            "question": "what is the length of the following text? 'Elon Reeve Musk FRS'"
        }
    )
    print(response)
    print(response['output'])


    # -------------------------------------------------------------------------------------------------------
    # Use the response
    # -------------------------------------------------------------------------------------------------------
    # Extract and parse the JSON
    # json_str = response['output'].strip('```json').strip('```').strip()
    # data = json.loads(json_str)

    # Access the data
    # print(data['answer'])
    # for source in data['sources']:
    #     print(f"Source URL: {source['url']}")


    # -------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # menu = CommandMenu({
    #     '1': function_1,
    # })
    function_1()
    # menu.run()
