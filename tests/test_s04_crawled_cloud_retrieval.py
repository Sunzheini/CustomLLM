import os
from pathlib import Path

from dotenv import load_dotenv

from support.callback_handler import CustomCallbackHandler


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


def test_11_retrieve_from_crawled_in_cloud(base_dir, managers):
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings(
        model="text-embedding-3-small",     # one of OpenAI's latest and most cost-effective embedding models
        show_progress_bar=True,             # display a progress bar
        chunk_size=50,                      # how many text strings are sent in a single batch request to the OpenAI API
        retry_min_seconds=10,               # wait at least 10 seconds before retrying the failed request
    )

    # 2
    pinecone_index_name = index2_name
    vectorstore = (managers['vector_store_manager']
    .get_vector_store(
        'pinecone', 'load',
        index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
    ))

    # 3
    query = "What is a LangChain Chain?"
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
    assert "LangChain" in response['answer']
    assert len(response['answer'].split('.')) >= 1  # Should have at least 1 sentence
    assert len(response['answer']) > 20  # Reasonable length for an answer

    print(f"\nAnswer: {response['answer']}")
