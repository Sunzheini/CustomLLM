import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import render_text_description
from support.callback_handler import CustomCallbackHandler

BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

open_ai_api_key = os.getenv('OPENAI_API_KEY')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
huggingface_hub = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def test_07_execute_llm_query(base_dir, managers):
    """
    Test executing a simple LLM query using a REACT agent with a text length tool.
    This test demonstrates the integration of various components including prompt
    templates, LLMs, and tools within a chain.
    :param base_dir: Base directory for the project
    :param managers: Dictionary of manager instances for prompts, LLMs, tools, and chains
    :return: None
    """
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 1
    tools = [managers['tools_manager'].get_text_length, ]

    # 3
    query = "what is the length of the following text? 'Elon Reeve Musk FRS'"
    react_prompt_template = managers['prompt_manager'].get_prompt_template("hwchase17/react")
    react_prompt = managers['prompt_manager'].prefill_existing_template(
        react_prompt_template,
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools])
    )

    # 4
    llm = managers['llm_manager'].get_llm("gpt-4.1-mini", temperature=0, callbacks=[CustomCallbackHandler()], bind_stop=True)

    # 5
    chain = managers['chains_manager'].get_react_agent_chain(llm, react_prompt, tools)

    # 6
    response = chain.invoke(input={"input": query})

    # ----------------------------------------------------------------------------------
    # Assert
    # ----------------------------------------------------------------------------------

    # 1. Basic response structure assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, dict), "Response should be a dictionary"
    assert 'output' in response, "Response should contain 'output' key"

    # 2. Check that the tool was used (intermediate steps should exist)
    assert 'intermediate_steps' in response, "Should have intermediate steps showing tool usage"
    assert len(response['intermediate_steps']) > 0, "Should have at least one intermediate step"

    # 3. Verify the final answer contains the correct length
    expected_length = 19  # Length of "Elon Reeve Musk FRS" without quotes
    output = response['output'].lower()

    # Check if the answer contains the correct length
    assert str(expected_length) in output, f"Output should contain the length {expected_length}. Got: {output}"

    # 4. Verify the response structure
    assert any(keyword in output for keyword in ['length', 'characters', 'count']), \
        f"Output should mention length/characters. Got: {output}"

    # 5. Optional: Check intermediate steps for tool usage
    first_step = response['intermediate_steps'][0]
    assert len(first_step) == 2, "Intermediate step should be a tuple of (action, observation)"

    action, observation = first_step
    assert hasattr(action, 'tool'), "Action should have a tool attribute"
    assert action.tool == 'get_text_length', f"Expected tool 'get_text_length', got '{action.tool}'"

    # 6. Verify the observation contains the correct length
    assert str(expected_length) in str(observation), \
        f"Tool observation should contain length {expected_length}. Got: {observation}"

    print(f"✓ Test passed! Response: {response['output']}")
