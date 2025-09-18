import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from langchain.agents import tool


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()


class ToolsManager:
    """
    Manages a collection of tools for use with language models. You can define your own
    tools here or import them from other modules.
    Tools allow llms to access external utilities e.g. search, calculations, etc.
    """
    def __init__(self):
        self.tools = {}

    @staticmethod
    @tool
    def get_text_length(text: Annotated[str, "The text to measure"]) -> int:
        """Returns the length of the input text."""
        print(f"get_text_length enter with {text=}")
        text = text.strip("'\n").strip('"')  # clean up the input text from extra quotes and newlines
        return len(text)

    @staticmethod
    @tool
    def triple(num: float) -> float:
        """
        Simple function to triple a number
        :param num: the number to triple
        :return: the tripled number as float
        """
        return float(num * 3)
