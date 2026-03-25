# Events
Communicate with agents via events.

Channel
https://www.youtube.com/@CopilotKit

Documentation
https://docs.ag-ui.com/introduction

Playground
https://dojo.ag-ui.com/


## Events
- RunStarted: Signals the start of an agent run
- StepStarted: Signals the start of a step within an agent run
- StepFinished: StepFinished
- RunFinished: Signals the successful completion of an agent run
- RunError: Signals an error during an agent run

- TextMessageStart: Signals the start of a text message
- TextMessageContent: Represents a chunk of content in a streaming text message
- TextMessageEnd: Signals the end of a text message
- TextMessageChunk: Convenience event that expands to Start → Content → End automatically

- ToolCallStart: Signals the start of a tool call
- ToolCallArgs: Represents a chunk of argument data for a tool call
- ToolCallEnd: Signals the end of a tool call
- ToolCallResult: Provides the result of a tool call execution
- ToolCallChunk: Convenience event that expands to Start → Args → End automatically

- StateSnapshot: Provides a complete snapshot of an agent’s state
- StateDelta: Provides a partial update to an agent’s state using JSON Patch
- MessagesSnapshot: Provides a snapshot of all messages in a conversation

- ActivitySnapshot: Delivers a complete snapshot of an activity message
- ActivityDelta: Applies incremental updates to an existing activity using JSON Patch operations

- Raw: Used to pass through events from external systems
- Custom: Used for application-specific custom events

- ReasoningStart: Marks the start of reasoning
- ReasoningMessageStart: Signals the start of a reasoning message
- ReasoningMessageContent: Represents a chunk of content in a streaming reasoning message
- ReasoningMessageEnd: Signals the end of a reasoning message
- ReasoningMessageChunk: A convenience event to auto start/close reasoning messages
- ReasoningEnd: Marks the end of reasoning
- ReasoningEncryptedValue: Attaches encrypted chain-of-thought reasoning to a message or tool call
