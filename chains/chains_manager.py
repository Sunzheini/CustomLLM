from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain.chains.base import Chain
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent


class ChainsManager:
    """Manages different types of chains for document processing and retrieval."""

    def __init__(self):
        pass

    @staticmethod
    def get_document_retrieval_chain(llm: BaseChatModel, prompt: BasePromptTemplate, vectorstore: VectorStore) -> Runnable:
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

    @staticmethod
    def get_react_agent_chain(llm: BaseChatModel, prompt: BasePromptTemplate, tools) -> Runnable:
        agent = create_react_agent(
            llm=llm,
            prompt=prompt,
            tools=tools,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,

            return_intermediate_steps=True,  # return intermediate steps for inspection

            handle_parsing_errors=True,
            max_iterations=5  # Add iteration limit for safety
        )

        react_agent_chain = agent_executor

        return react_agent_chain

    @staticmethod
    def get_document_retrieval_chain_with_history(llm: BaseChatModel, answer_prompt: BasePromptTemplate, history_prompt: BasePromptTemplate, vectorstore: VectorStore) -> Runnable:
        """
        Create a retrieval chain using the provided vectorstore and LLM.

        Args:
            llm: The language model to use
            answer_prompt: The prompt template for generating answers
            history_prompt: The prompt template for incorporating chat history
            vectorstore: The vector store for retrieval

        Returns:
            A configured retrieval chain

        Raises:
            ValueError: If any required parameter is None
        """
        if llm is None or answer_prompt is None or history_prompt is None or vectorstore is None:
            raise ValueError("LLM, prompt and vectorstore must be provided")

        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            prompt=history_prompt
        )

        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=answer_prompt
        )

        retrieval_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain
        )

        return retrieval_chain
