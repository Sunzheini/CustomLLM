import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()


class EmbeddingsManager:
    """ Manages a collection of embeddings for different models. """
    def __init__(self):
        self.__open_ai_embeddings = OpenAIEmbeddings

    @property
    def open_ai_embeddings(self):
        return self.__open_ai_embeddings
