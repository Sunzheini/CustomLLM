import os
from copy import deepcopy
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()


class LlmManager:
    """
    Manages the LLM instance with specific configurations.

    """
    def __init__(self):
        """
        Initialize the LLM manager with a specific LLM instance.
        """
        self.llm_configs = {
            # small model
            'gemma3:270m': {"class": ChatOllama, "params": {"temperature": 0, "model": "gemma3:270m"}},
            # large model, don't run until 32GB RAM
            'gpt-oss:20b': {"class": ChatOllama, "params": {"temperature": 0, "model": "gpt-oss:20b"}},
            # medium, slow
            'gemma3:4b': {"class": ChatOllama, "params": {"temperature": 0, "model": "gemma3:4b"}},
            # 3
            'gpt-4.1-mini': {"class": ChatOpenAI, "params": {"temperature": 0, "model": "gpt-4.1-mini"}}
        }

    def get_llm(self, llm_name: str, temperature: int, callbacks: Optional[List[BaseCallbackHandler]] = None, bind_stop: bool = False) -> BaseChatModel:
        """
        Factory method to get LLM instances with optional callbacks.
        bind_stop: Safety net, not always triggered, stop sequence to end generation when the model outputs anything from the list

        Args:
            llm_name: Name of the LLM configuration
            temperature: Sampling temperature for the model, 0 for deterministic, 1 for creative
            callbacks: List of callback handlers (None for no callbacks)
            bind_stop: Whether to bind stop sequences (mainly for OpenAI models)

        Returns:
            Configured LLM instance
        """
        if llm_name not in self.llm_configs:
            raise ValueError(f"LLM '{llm_name}' is not supported. Available: {list(self.llm_configs.keys())}")

        config = self.llm_configs[llm_name]
        llm_class = config["class"]

        # Use deepcopy to create a safe copy of parameters
        params = deepcopy(config["params"])
        params["temperature"] = temperature

        if callbacks:
            params["callbacks"] = callbacks

        # Instantiate + Bind stop sequences only if requested AND for OpenAI models
        if bind_stop and llm_name == 'gpt-4.1-mini':
            llm = llm_class(**params).bind(stop=["\nObservation:", "Observation"])

        # for other models, just instantiate
        else:
            llm = llm_class(**params)

        return llm
