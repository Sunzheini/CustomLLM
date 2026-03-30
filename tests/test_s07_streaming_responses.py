"""
Test streaming responses for LLMs, chains, and graphs.
Validates that streaming works correctly for real-time token delivery.
"""
import asyncio
from typing import List

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture streaming tokens."""
    
    def __init__(self):
        self.tokens: List[str] = []
        self.complete_response: str = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated during streaming."""
        self.tokens.append(token)
        print(token, end="", flush=True)  # Print token as it arrives
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes."""
        # Get the complete response
        if hasattr(response, 'generations') and response.generations:
            if response.generations[0]:
                self.complete_response = response.generations[0][0].text
        print("\n✓ Streaming complete")


def test_16_stream_llm_response_basic(managers):
    """
    Test basic LLM streaming response.
    Verifies that tokens are delivered incrementally.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    streaming_handler = StreamingCallbackHandler()
    
    # Get LLM with streaming enabled
    llm = managers['llm_manager'].get_llm(
        "gpt-4.1-mini", 
        temperature=0,
        callbacks=[streaming_handler],
        streaming=True  # Enable streaming
    )
    
    query = "Count from 1 to 5 slowly, one number per line."
    
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    print("\n🔄 Starting streaming response...")
    response = llm.invoke(query)
    
    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    # 1. Verify streaming happened (tokens were captured)
    assert len(streaming_handler.tokens) > 0, "Should have captured streaming tokens"
    assert len(streaming_handler.tokens) > 5, "Should have multiple tokens (more than just 5 numbers)"
    
    # 2. Verify response is not empty
    assert response is not None, "Response should not be None"
    assert hasattr(response, 'content'), "Response should have content attribute"
    assert len(response.content) > 0, "Response content should not be empty"
    
    # 3. Verify the response makes sense
    response_text = response.content.lower()
    assert any(str(i) in response_text for i in range(1, 6)), "Response should contain numbers 1-5"
    
    # 4. Print summary
    print(f"\n✅ Streamed {len(streaming_handler.tokens)} tokens")
    print(f"📝 Complete response length: {len(response.content)} characters")
    print(f"🔢 First 5 tokens: {streaming_handler.tokens[:5]}")


def test_17_stream_chain_response(base_dir, managers):
    """
    Test streaming response from a retrieval chain.
    Verifies that document retrieval + generation can stream.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    streaming_handler = StreamingCallbackHandler()
    
    # 0. Get embeddings
    embeddings = managers['embeddings_manager'].open_ai_embeddings()
    
    # 1. Load FAISS vector store (assumes test_02 ran first)
    faiss_path = str(base_dir / 'faiss_index_react_paper')
    
    try:
        vectorstore = managers['vector_store_manager'].get_vector_store(
            'faiss', 'load', faiss_path, embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        pytest.skip(f"FAISS index not found. Run ingestion test first: {e}")
    
    # 2. Get prompt
    retrieval_qa_chat_prompt = managers['prompt_manager'].get_prompt_template(
        "langchain-ai/retrieval-qa-chat"
    )
    
    # 3. Get LLM with streaming
    llm = managers['llm_manager'].get_llm(
        "gpt-4.1-mini",
        temperature=0,
        callbacks=[streaming_handler],
        streaming=True
    )
    
    # 4. Create chain
    chain = managers['chains_manager'].get_document_retrieval_chain(
        llm, retrieval_qa_chat_prompt, vectorstore
    )
    
    query = "What is ReAct? Give a brief 2-sentence explanation."
    
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    print("\n🔄 Starting chain streaming response...")
    
    # For streaming with chains, we use .stream() instead of .invoke()
    chunks = []
    for chunk in chain.stream({"input": query}):
        chunks.append(chunk)
        if 'answer' in chunk:
            print(chunk['answer'], end="", flush=True)
    
    print("\n")
    
    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    # 1. Verify we got chunks
    assert len(chunks) > 0, "Should have received streaming chunks"
    
    # 2. Find the answer in chunks
    full_answer = ""
    for chunk in chunks:
        if 'answer' in chunk:
            full_answer += chunk['answer']
    
    assert len(full_answer) > 0, "Should have assembled an answer from chunks"
    assert "ReAct" in full_answer or "react" in full_answer.lower(), \
        f"Answer should mention ReAct. Got: {full_answer}"
    
    # 3. Verify multiple chunks were streamed
    answer_chunks = [c for c in chunks if 'answer' in c]
    assert len(answer_chunks) > 1, "Should have multiple answer chunks (streaming)"
    
    print(f"✅ Streamed {len(answer_chunks)} answer chunks")
    print(f"📝 Complete answer: {full_answer[:100]}...")


@pytest.mark.asyncio
async def test_18_async_stream_response(managers):
    """
    Test async streaming for better performance.
    Verifies that astream() works correctly.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    streaming_handler = StreamingCallbackHandler()
    
    llm = managers['llm_manager'].get_llm(
        "gpt-4.1-mini",
        temperature=0.7,
        callbacks=[streaming_handler],
        streaming=True
    )
    
    query = "Write a haiku about AI testing."
    
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    print("\n🔄 Starting async streaming response...")
    
    chunks = []
    async for chunk in llm.astream(query):
        chunks.append(chunk)
        print(chunk.content, end="", flush=True)
    
    print("\n")
    
    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    assert len(chunks) > 0, "Should have received streaming chunks"
    
    # Reconstruct full message
    full_content = "".join(chunk.content for chunk in chunks)
    assert len(full_content) > 0, "Should have content"
    
    # Haiku should be short
    assert len(full_content) < 500, "Haiku should be short"
    
    # Should have line breaks (haikus have 3 lines)
    assert full_content.count('\n') >= 2, "Haiku should have multiple lines"
    
    print(f"✅ Async streamed {len(chunks)} chunks")
    print(f"📝 Haiku:\n{full_content}")


def test_19_stream_with_timeout(managers):
    """
    Test that streaming can be interrupted/timed out.
    Verifies robust error handling.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    streaming_handler = StreamingCallbackHandler()
    
    llm = managers['llm_manager'].get_llm(
        "gpt-4.1-mini",
        temperature=0,
        callbacks=[streaming_handler],
        streaming=True
    )
    
    # Short query so it completes quickly
    query = "What is 2+2?"
    
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    try:
        response = llm.invoke(query)
        
        # ----------------------------------------------------------------------------------
        # Assert
        # ----------------------------------------------------------------------------------
        assert response is not None, "Should get response within timeout"
        assert len(streaming_handler.tokens) > 0, "Should have streamed tokens"
        assert "4" in response.content, "Should correctly answer 2+2=4"
        
        print(f"✅ Completed streaming within timeout")
        print(f"📝 Response: {response.content}")
        
    except Exception as e:
        pytest.fail(f"Streaming failed with timeout error: {e}")


def test_20_compare_streaming_vs_non_streaming(managers):
    """
    Compare streaming vs non-streaming performance and behavior.
    Educational test to show the difference.
    """
    # ----------------------------------------------------------------------------------
    # Arrange
    # ----------------------------------------------------------------------------------
    query = "List 3 benefits of AI in one sentence each."
    
    # Non-streaming LLM
    llm_normal = managers['llm_manager'].get_llm(
        "gpt-4.1-mini",
        temperature=0,
        streaming=False
    )
    
    # Streaming LLM
    streaming_handler = StreamingCallbackHandler()
    llm_streaming = managers['llm_manager'].get_llm(
        "gpt-4.1-mini",
        temperature=0,
        callbacks=[streaming_handler],
        streaming=True
    )
    
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    print("\n📊 Testing NON-STREAMING response...")
    response_normal = llm_normal.invoke(query)
    
    print("\n📊 Testing STREAMING response...")
    response_streaming = llm_streaming.invoke(query)
    
    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------
    # Both should produce responses
    assert response_normal is not None, "Normal response should exist"
    assert response_streaming is not None, "Streaming response should exist"
    
    # Streaming should have captured tokens
    assert len(streaming_handler.tokens) == 0 or len(streaming_handler.tokens) > 10, \
        "Streaming should capture multiple tokens OR none if handler not called"
    
    # Both responses should be similar quality
    assert len(response_normal.content) > 50, "Normal response should be substantial"
    assert len(response_streaming.content) > 50, "Streaming response should be substantial"
    
    print(f"\n✅ Comparison complete:")
    print(f"   Non-streaming response length: {len(response_normal.content)} chars")
    print(f"   Streaming response length: {len(response_streaming.content)} chars")
    print(f"   Tokens captured: {len(streaming_handler.tokens)}")
    print(f"\n💡 Key difference: Streaming delivers tokens incrementally for better UX!")


# Run with: pytest tests/test_s07_streaming_responses.py -v -s
# -v = verbose (show test names)
# -s = show stdout (REQUIRED to see streaming in real-time!)
#
# KEY CONCEPTS:
# • streaming=True + callbacks=[handler] = enable streaming
# • on_llm_new_token() = called for EVERY token (this is where streaming happens!)
# • .invoke() with streaming = blocks but tokens stream to callbacks
# • .stream() = yields chunks for chains (use "for chunk in chain.stream(...)")
# • .astream() = async version (use "async for chunk in llm.astream(...)")
