import time

import streamlit as st


st.header ("CustomLLM Project")

prompt = st.text_input(
    "Prompt",                           # label
    placeholder="Enter your prompt:"    # placeholder text
)

button = st.button("Go")


if 'prompt_history' not in st.session_state:
    st.session_state['prompt_history'] = []

if 'answer_history' not in st.session_state:
    st.session_state['answer_history'] = []

if 'processing' not in st.session_state:
    st.session_state['processing'] = False


def generate_llm_response(user_prompt):
    time.sleep(1)  # Simulate processing time
    return f"Response to: '{user_prompt}'\n\nThis is ..."


# Handle button click and prompt submission
if button and prompt:
    st.session_state['processing'] = True

    with st.spinner("Processing..."):
        try:
            generated_response = generate_llm_response(prompt)

            st.session_state['prompt_history'].append(prompt)
            st.session_state['answer_history'].append(generated_response)

            st.success("Done!")
            st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            st.session_state['processing'] = False


# Also handle Enter key in text input, instead of only button click
if prompt and not button and not st.session_state['processing']:
    st.session_state['processing'] = True

    with st.spinner("Processing..."):
        try:
            generated_response = generate_llm_response(prompt)

            st.session_state['prompt_history'].append(prompt)
            st.session_state['answer_history'].append(generated_response)

            st.success("Done!")
            st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

        finally:
            st.session_state['processing'] = False


if st.session_state['answer_history']:
    for generated_response, prompt in zip(st.session_state['answer_history'], st.session_state['prompt_history']):
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write(generated_response)
