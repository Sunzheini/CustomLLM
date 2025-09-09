import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from core.command_menu import CommandMenu
from support.measure_and_print_time_decorator import measure_and_print_time_decorator


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

open_ai_api_key = os.getenv('OPENAI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')


@measure_and_print_time_decorator
def function_1():
    info = """
    Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion.
    Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he had obtained Canadian citizenship at birth through his Canadian-born mother. He received bachelor's degrees in 1997 from the University of Pennsylvania in Philadelphia, United States, before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became an American citizen.
    """

    summary_template = """
    Given the information {info} about a person I want you to create:
    1. Short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["info"],
        template=summary_template
    )

    # temperature=0 for deterministic, 1 for creative
    # llm = ChatOpenAI(temperature=0, model="gpt-4.1-mini")
    llm = ChatOllama(temperature=0, model="gemma3:270m")        # small model
    # llm = ChatOllama(temperature=0, model="gpt-oss:20b")        # large model, dont run until 32GB RAM
    # llm = ChatOllama(temperature=0, model="gemma3:4b")          # medium, slow

    chain = summary_prompt_template | llm   # output of one is input to the next

    response = chain.invoke(input={
        "info": info
    })

    print(response.content)


# ------------------------------------------------------------------------------


if __name__ == "__main__":
    menu = CommandMenu({
        '1': function_1,
    })
    menu.run()
