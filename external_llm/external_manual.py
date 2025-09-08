from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class ExternalManual:
    def __init__(self, llm):
        self.llm = llm
        self.external_api_url = None

    def check_connection(self):
        pass

    def send_request(self, messages):
        if not self.check_connection():
            raise ConnectionError("Cannot connect to the external LLM service.")


        pass
