from support.callback_handler import CustomCallbackHandler


def test_03_retrieve_from_pdf_in_local(base_dir, managers):
    """
    Test the complete RAG pipeline for PDF retrieval.

    This integration test verifies that:
    1. FAISS vector store can be loaded
    2. Prompt retrieval works
    3. LLM can be initialized with callbacks
    4. Retrieval chain can be created
    5. The chain produces a meaningful response containing "ReAct"

    Note: Requires pre-created FAISS index from ingestion process.
    """
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings()

    # 2
    faiss_path = str(base_dir / 'faiss_index_react_paper')
    vectorstore = managers['vector_store_manager'].get_vector_store('faiss', 'load', faiss_path, embeddings, allow_dangerous_deserialization=True)

    # 3
    query = "Give me the gist of ReAct in 3 sentences."
    # query = "When was the ReAct paper published?"
    retrieval_qa_chat_prompt = managers['prompt_manager'].get_prompt_template("langchain-ai/retrieval-qa-chat")

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

    # 5
    chain = managers['chains_manager'].get_document_retrieval_chain(llm, retrieval_qa_chat_prompt, vectorstore)

    # 6
    response = chain.invoke(input={"input": query})

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    assert "ReAct" in response['answer']
    assert len(response['answer'].split('.')) >= 3      # Should have at least 3 sentences
    assert len(response['answer']) > 20                 # Reasonable length for an answer

    print(f"\nAnswer: {response['answer']}")
