CUSTOM_USER_PROMPT = """
    Given the information {question} about a person I want you to create:
    1. Short summary
    2. Two interesting facts about them
    
    You have access to the following tools: {tools}
    
    Action: the action to take, should be one of [{tool_names}]
    Final Answer: the final answer to the original input question formatted as JSON according to these format instructions: {format_instructions}
    
    Thought:{agent_scratchpad}
"""
