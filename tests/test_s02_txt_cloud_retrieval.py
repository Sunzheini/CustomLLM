import os

from dotenv import load_dotenv

from support.callback_handler import CustomCallbackHandler


load_dotenv()


pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')


def test_06_retrieve_from_txt_in_cloud(base_dir, managers):
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings()

    # 2
    pinecone_index_name = index_name
    vectorstore = (managers['vector_store_manager']
    .get_vector_store(
        'pinecone', 'load',
        index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
    ))

    # 3
    query = "Is this text related to Wikipedia?"
    # query = "Has Evan Chaki published any articles on vector databases?"
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
    assert "Wikipedia" in response['answer']
    assert len(response['answer'].split('.')) >= 1  # Should have at least 1 sentence
    assert len(response['answer']) > 10  # Reasonable length for an answer

    print(f"\nAnswer: {response['answer']}")
