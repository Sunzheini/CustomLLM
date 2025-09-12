import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import tool


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()


class ToolsManager:
    """
    Manages a collection of tools for use with language models. You can define your own
    tools here or import them from other modules.
    """
    def __init__(self):
        self.tools = {}

    @tool
    def get_text_length(self, text: str) -> int:
        """Returns the length of the input text."""     # llm uses this description to decide when to use this tool
        print(f"get_text_length enter with {text=}")
        text = text.strip("'\n").strip(
            '"'
        )   # clean up the input text from extra quotes and newlines
        return len(text)
