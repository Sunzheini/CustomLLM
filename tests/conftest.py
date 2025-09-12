from pathlib import Path

import pytest

from communication.communications_manager import CommunicationsManager
from context.chains.chains_manager import ChainsManager
from embeddings.embeddings_manager import EmbeddingsManager
from llm_manager import LlmManager
from prompts.prompt_manager import PromptManager
from tools.tools_manager import ToolsManager
from vector_stores.vector_store_manager import VectorStoreManager


@pytest.fixture(scope="session")
def base_dir():
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def managers():
    return {
        'embeddings_manager': EmbeddingsManager(),              # 0
        'tools_manager': ToolsManager(),                        # 1
        'vector_store_manager': VectorStoreManager(),           # 2
        'prompt_manager': PromptManager(),                      # 3
        'llm_manager': LlmManager(),                            # 4
        'chains_manager': ChainsManager(),                      # 5
        'communications_manager': CommunicationsManager(),      # 6
    }






# run with `pytest` in th terminal
# run with `pytest -s` to see print statements
