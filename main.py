import os
from random import randint
from pathlib import Path

from dotenv import load_dotenv
from ollama_local_llm.ollama_manual import OllamaManual
from ollama_local_llm.ollama_automatic import OllamaAutomatic

from core.command_menu import CommandMenu
from support.measure_and_print_time_decorator import measure_and_print_time_decorator


BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file (for local development)
if os.path.exists(os.path.join(BASE_DIR, '.env')):
    load_dotenv()

open_ai_api_key = os.getenv('OPEN_AI_API_KEY')
print(open_ai_api_key)


@measure_and_print_time_decorator
def function_1():
    """Ollama manual HTTP requests with streaming"""
    ollama_manual = OllamaManual(model="codellama")

    num = randint(1, 10)
    messages = [
        {
            "role": "user",
            "content": f"What is 1 + {num}? Please explain step by step.",
        }
    ]

    response = ollama_manual.send_request(messages)
    if response:
        print(f"\nFull response received: {len(response)} characters")


@measure_and_print_time_decorator
def function_2():
    """Ollama using official client - Basic generation"""
    ollama_client = OllamaAutomatic(model="codellama")

    num = randint(1, 10)
    prompt = f"What is 1 + {num}? Please explain step by step."

    response = ollama_client.generate_response(prompt)
    if response:
        print("\nResponse from Ollama (official client):")
        print(response)
        print(f"\nFull response length: {len(response)} characters")


@measure_and_print_time_decorator
def function_3():
    """Ollama using official client - Streaming generation"""
    ollama_client = OllamaAutomatic(model="codellama")

    num = randint(1, 10)
    prompt = f"What is 1 + {num}? Please explain step by step."

    print("Streaming response:")
    full_response = ""
    for chunk in ollama_client.stream_generate(prompt):
        if chunk and isinstance(chunk, str):
            full_response += chunk

    print(f"\nFull streamed response length: {len(full_response)} characters")


@measure_and_print_time_decorator
def function_4():
    """Ollama using official client - Chat with history"""
    ollama_client = OllamaAutomatic(model="codellama")

    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user", "content": "Now multiply that by 3."}
    ]

    # Print the entire conversation first
    print("Full conversation history:")
    for i, msg in enumerate(messages):
        print(f"{i + 1}. {msg['role'].upper()}: {msg['content']}")

    print("\n--- Generating next response ---")

    response = ollama_client.chat(messages)
    if response:
        # Add the new response to the conversation
        messages.append({"role": "assistant", "content": response})

        print("\nUpdated conversation:")
        for i, msg in enumerate(messages):
            print(f"{i + 1}. {msg['role'].upper()}: {msg['content']}")


if __name__ == "__main__":
    menu = CommandMenu({
        '1': function_1,
        '2': function_2,
        '3': function_3,
        '4': function_4,
    })
    menu.run()
