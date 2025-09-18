import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain import hub

from langchain_core.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate

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
            },
            "hwchase17/react": {
                "type": "hub",
                "link": "https://smith.langchain.com/hub/hwchase17/react?organizationId=4d2f1613-26c5-4bb8-b70c-40b7f844b650"
            },

            "langchain-ai/chat-langchain-rephrase": {
                "type": "hub",
                "link": "https://smith.langchain.com/hub/langchain-ai/chat-langchain-rephrase?organizationId=4d2f1613-26c5-4bb8-b70c-40b7f844b650"
            },

            "custom_prompt1": {
                "type": "local",
                "path": os.path.join(BASE_DIR, 'custom_prompt1.py')
            },
            "custom_prompt2": {
                "type": "local",
                "path": os.path.join(BASE_DIR, 'custom_prompt2.py')
            },
        }

    @staticmethod
    def __detect_input_variables(template: str) -> List[str]:
        """Simple method to detect input variables in template string"""
        import re
        # This is a simple regex to find {variable} patterns
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))  # Remove duplicates

    def get_prompt_template(self, prompt_name: str) -> BasePromptTemplate:
        """
        Create a prompt based on the input.
        :param prompt_name: The name of the prompt to create
        :return: The created prompt as a PromptTemplate object
        """
        if prompt_name not in self.__prompt_library:
            raise ValueError(f"Prompt '{prompt_name}' not found in the prompt library.")

        prompt = self.__prompt_library[prompt_name]

        if prompt["type"] == "hub":
            return hub.pull(prompt_name)

        elif prompt["type"] == "local":
            try:
                with open(prompt["path"], 'r', encoding='utf-8') as f:
                    template_content = f.read().strip()
                return PromptTemplate(
                    template=template_content,
                    input_variables=self.__detect_input_variables(template_content)
                )

            except FileNotFoundError:
                raise ValueError(f"Prompt file not found: {prompt['path']}")

            except Exception as e:
                raise ValueError(f"Error loading prompt from file: {e}")

        else:
            raise ValueError(f"Unknown prompt type '{prompt['type']}' for prompt '{prompt_name}'.")

    def list_prompts(self) -> list:
        """Return a list of available prompt names"""
        return list(self.__prompt_library.keys())

    @staticmethod
    def prefill_existing_template(template: BasePromptTemplate, **kwargs) -> BasePromptTemplate:
        """
        Prefill an existing prompt template with provided arguments.
        Now accepts PromptTemplate objects instead of strings.

        :param template: The PromptTemplate object to prefill
        :param kwargs: Keyword arguments for the template
        :return: The filled prompt template
        """
        # Use the partial method directly on the PromptTemplate object
        return template.partial(**kwargs)

    @staticmethod
    def create_template_from_messages(messages) -> ChatPromptTemplate:
        template = ChatPromptTemplate.from_messages(messages)
        return template
