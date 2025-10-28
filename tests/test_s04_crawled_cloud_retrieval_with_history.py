import os

from dotenv import load_dotenv

from support.callback_handler import CustomCallbackHandler


load_dotenv()


pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
index2_name = os.getenv('INDEX2_NAME')


def test_12_retrieve_from_crawled_in_cloud_with_history(base_dir, managers):
    prompts = {
        'prompt1': "What is LangChain?",
        'prompt2': "What did I just ask you?",
        'prompt3': "How many letters are in your first reply to me?",
    }
    chat_history = []  # list of (user, bot) tuples

    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    # 0
    embeddings = managers['embeddings_manager'].open_ai_embeddings(
        model="text-embedding-3-small",
        show_progress_bar=True,
        chunk_size=50,
        retry_min_seconds=10,
    )

    # 2
    pinecone_index_name = index2_name
    vectorstore = (managers['vector_store_manager']
    .get_vector_store(
        'pinecone', 'load',
        index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
    ))

    # 3
    query = None
    retrieval_qa_chat_prompt = managers['prompt_manager'].get_prompt_template("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = managers['prompt_manager'].get_prompt_template("langchain-ai/chat-langchain-rephrase")

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()])

    # 5
    chain = managers['chains_manager'].get_document_retrieval_chain_with_history(
        llm,
        retrieval_qa_chat_prompt,
        rephrase_prompt,
        vectorstore
    )

    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    responses = {}

    for key, prompt in prompts.items():
        query = prompt

        print(f'before request {key}: {query}')

        # 6
        response = chain.invoke(input={"input": query, "chat_history": chat_history})

        # ------------------------------------------------------------------------------
        # Assertions for each response
        # ------------------------------------------------------------------------------
        assert 'answer' in response, f"Response should contain 'answer' key for {key}"
        assert isinstance(response['answer'], str), f"Answer should be a string for {key}"
        assert len(response['answer'].strip()) > 0, f"Answer should not be empty for {key}"

        # Store response for later assertions
        responses[key] = response['answer']

        # ------------------------------------------------------------------------------
        chat_history.append(('human', query))
        chat_history.append(('ai', response['answer']))

        print(f'after request {key}: {response["answer"]}')
        print('-' * 10)

    # ----------------------------------------------------------------------------------
    # Additional Assertions
    # ----------------------------------------------------------------------------------
    # Test 1: First response should contain LangChain information
    assert 'langchain' in responses['prompt1'].lower(), "First response should mention LangChain"
    assert any(keyword in responses['prompt1'].lower() for keyword in ['framework', 'library', 'tool']), \
        "First response should describe LangChain as a framework/library/tool"

    # Test 2: Second response should reference the first question
    assert 'langchain' in responses['prompt2'].lower() or 'what is' in responses['prompt2'].lower(), \
        "Second response should reference the first question about LangChain"

    # Test 3: Third response should attempt to answer the counting question
    assert any(char.isdigit() for char in responses['prompt3']), \
        "Third response should contain a number (letter count)"

    # Test 4: Chat history should be properly maintained
    assert len(chat_history) == 6, f"Chat history should have 6 entries, got {len(chat_history)}"
    assert chat_history[0] == ('human', 'What is LangChain?'), "First chat history entry incorrect"
    assert chat_history[1][0] == 'ai', "Second chat history entry should be AI response"
    assert chat_history[2] == ('human', 'What did I just ask you?'), "Third chat history entry incorrect"
    assert chat_history[3][0] == 'ai', "Fourth chat history entry should be AI response"
    assert chat_history[4] == ('human',
                               'How many letters are in your first reply to me?'), "Fifth chat history entry incorrect"
    assert chat_history[5][0] == 'ai', "Sixth chat history entry should be AI response"

    # Test 5: Responses should be different (not just repeating the same answer)
    assert responses['prompt1'] != responses['prompt2'] != responses['prompt3'], \
        "Responses to different questions should be different"

    # Test 6: Verify response lengths are reasonable
    assert len(responses['prompt1']) > 50, "First response should be detailed"
    assert len(responses['prompt2']) > 20, "Second response should be meaningful"
    assert len(responses['prompt3']) > 10, "Third response should be meaningful"

    print("✅ All tests passed! Chat history functionality is working correctly.")
    print(f"Final chat history length: {len(chat_history)} entries")
    for i, (speaker, message) in enumerate(chat_history):
        print(f"{i}: {speaker}: {message[:100]}{'...' if len(message) > 100 else ''}")
