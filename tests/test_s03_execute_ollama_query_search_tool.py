import os
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import render_text_description

from support.callback_handler import CustomCallbackHandler


BASE_DIR = Path(__file__).resolve().parent

if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')


def test_08_execute_ollama_query_search_tool(base_dir, managers):
    # ----------------------------------------------------------------------------------
    # Act
    # ----------------------------------------------------------------------------------
    # 1
    tools = [TavilySearchResults(api_key=tavily_api_key, max_results=3)]

    # 3
    query = "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    react_prompt_template = managers['prompt_manager'].get_prompt_template("hwchase17/react")
    react_prompt = managers['prompt_manager'].prefill_existing_template(
        react_prompt_template,
        tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]),
    )

    # 4
    llm = managers['llm_manager'].get_llm("gemma3:4b", temperature=0, callbacks=[CustomCallbackHandler()], bind_stop=True)

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

    # 3. Verify the search tool was called
    first_step = response['intermediate_steps'][0]
    assert len(first_step) == 2, "Intermediate step should be a tuple of (action, observation)"

    action, observation = first_step
    assert hasattr(action, 'tool'), "Action should have a tool attribute"

    # Handle different possible tool names
    expected_tool_names = ['tavily_search_results_json', 'tavily_search_results', 'search']
    assert action.tool in expected_tool_names, f"Expected one of {expected_tool_names}, got '{action.tool}'"

    # 4. Handle tool_input which might be a string or dict
    assert hasattr(action, 'tool_input'), "Action should have tool_input attribute"

    search_query = None
    if isinstance(action.tool_input, dict):
        # If tool_input is a dictionary, extract the query
        assert 'query' in action.tool_input, "Tool input dict should contain 'query' key"
        search_query = action.tool_input['query'].lower()
    elif isinstance(action.tool_input, str):
        # If tool_input is a string, it might be the query itself or JSON
        try:
            # Try to parse as JSON
            parsed_input = json.loads(action.tool_input)
            if isinstance(parsed_input, dict) and 'query' in parsed_input:
                search_query = parsed_input['query'].lower()
            else:
                search_query = action.tool_input.lower()
        except json.JSONDecodeError:
            # If not JSON, assume it's the query string
            search_query = action.tool_input.lower()
    else:
        # For other types, convert to string
        search_query = str(action.tool_input).lower()

    # 5. Verify the search query was appropriate
    relevant_keywords = ['ai engineer', 'langchain', 'bay area', 'linkedin', 'job', 'san francisco']
    assert any(keyword in search_query for keyword in relevant_keywords), \
        f"Search query should contain relevant keywords. Query: {search_query}"

    # 6. Verify the observation contains search results
    assert observation is not None, "Observation should not be None"
    assert isinstance(observation, (str, list, dict)), "Observation should be a string, list, or dict"

    # 7. Verify the final output contains job-related information
    output = response['output'].lower()

    # Check for job-related content in the output
    job_keywords = ['job', 'position', 'engineer', 'ai', 'langchain', 'linkedin', 'bay area', 'san francisco', 'hire']
    assert any(keyword in output for keyword in job_keywords), \
        f"Output should contain job-related information. Got: {output}"

    # 8. Verify the output structure suggests search was performed
    # Look for indicators that search results were processed
    search_indicators = ['found', 'result', 'search', 'according to', 'based on', 'showing']
    assert any(indicator in output for indicator in search_indicators) or \
           any(str(i) in output for i in range(1, 4)), \
        f"Output should indicate search results were processed. Got: {output}"

    # 9. Verify the response is comprehensive (not just a single line or error)
    assert len(output.strip()) > 50, "Output should be comprehensive (more than 50 characters)"
    assert not output.startswith(('error', 'fail', 'sorry')), f"Output suggests an error: {output}"

    # 10. Verify the response format is reasonable
    assert not (output.startswith('{') and output.endswith('}')), \
        "Output should not be raw JSON data"
    assert not (output.startswith('[') and output.endswith(']')), \
        "Output should not be raw array data"

    print(f"✓ Test passed! Response contains search results for AI engineer jobs.")
    print(f"Search query used: {search_query}")
    print(f"Output length: {len(output)} characters")
    print(f"Tool used: {action.tool}")
    print(f"Tool input type: {type(action.tool_input)}")
