import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


@tool
def triple(num: float) -> float:
    """
    Simple function to triple a number
    :param num: the number to triple
    :return: the tripled number as float
    """
    return float(num * 3)


tools = [TavilySearch(max_results=1), triple]

"""
function calling LLM with tools: the vendor is responsible 
to call the tools and handles the parsing of the response
"""
llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0).bind_tools(tools)     # function calling
