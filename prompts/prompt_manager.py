import os
from pathlib import Path

from dotenv import load_dotenv
from langchain import hub

from langchain_core.prompts import BasePromptTemplate


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()


class PromptManager:
    """
    A class to manage and store prompts for various tasks.
    """
    def __init__(self):
        self.__prompt_library = {
            "langchain-ai/retrieval-qa-chat": {
                "type": "hub",
                "link": "https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat?organizationId=4d2f1613-26c5-4bb8-b70c-40b7f844b650"
            }
        }

    def get_prompt(self, prompt_name: str):
        """
        Create a prompt based on the input.
        :param prompt_name: The name of the prompt to create
        :return: The created prompt as a string
        """
        if prompt_name not in self.__prompt_library:
            raise ValueError(f"Prompt '{prompt_name}' not found in the prompt library.")

        prompt = self.__prompt_library[prompt_name]

        if prompt["type"] == "hub":
            return hub.pull(prompt_name)

        elif prompt["type"] == "template":
            return None
        else:
            raise ValueError(f"Unknown prompt type '{prompt['type']}' for prompt '{prompt_name}'.")

    def list_prompts(self) -> list:
        """Return a list of available prompt names"""
        return list(self.__prompt_library.keys())
