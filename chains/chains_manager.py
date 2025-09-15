from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain


class ChainsManager:
    """Manages different types of chains for document processing and retrieval."""

    def __init__(self):
        pass

    @staticmethod
    def get_pdf_retrieval_chain(
            llm: BaseChatModel,
            prompt: BasePromptTemplate,
            vectorstore: VectorStore
    ) -> Chain:
        """
        Create a retrieval chain using the provided vectorstore and LLM.

        Args:
            llm: The language model to use
            prompt: The prompt template for document processing
            vectorstore: The vector store for retrieval

        Returns:
            A configured retrieval chain

        Raises:
            ValueError: If any required parameter is None
        """
        if llm is None or prompt is None or vectorstore is None:
            raise ValueError("LLM, prompt and vectorstore must be provided")

        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(),
            combine_docs_chain=combine_docs_chain
        )

        return retrieval_chain
