import os
from pathlib import Path

from dotenv import load_dotenv

from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from langgraph_example.react import tools, llm


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    :param state: The current state of messages.
    :return: The updated state after reasoning.
    """
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state['messages']]
    )
    return {'messages': [response]}     # if the results look liek this, then it is appended to the message state


tool_node = ToolNode(tools)
