import asyncio
import time

import streamlit as st


class StreamlitFrontend:
    def __init__(self, backend_object):
        self.backend = backend_object
        self._initialize_frontend()

    def _generate_llm_response(self, user_prompt, chat_history):
        response = self.backend.generate_response(user_prompt, chat_history=chat_history)
        return response

    def _initialize_frontend(self):
        st.header ("CustomLLM Project")

        prompt = st.text_input(
            "Prompt",                           # label
            placeholder="Enter your prompt:"    # placeholder text
        )

        # Buttons ----------------------------------------------------------
        row1_col1, row1_col2, = st.columns([1, 2])  # 2nd takes 2x space
        with row1_col1:
            button1 = st.button("Ingest")

        with row1_col2:
            button2 = st.button("Clean")

        row2_col1, row2_col2 = st.columns([2, 2])
        with row2_col1:
            button3 = st.button("Ask")

        # State management ------------------------------------------------
        if 'prompt_history' not in st.session_state:
            st.session_state['prompt_history'] = []

        if 'answer_history' not in st.session_state:
            st.session_state['answer_history'] = []

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []   # (user, bot) tuples

        if 'processing' not in st.session_state:
            st.session_state['processing'] = False

        # Handles ----------------------------------------------------
        # Handle button2, ingest data
        if button1:
            with st.spinner("Ingesting data..."):
                try:
                    asyncio.run(self.backend.crawled_cloud_async_ingestion())

                    st.success("Ingestion complete!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

        # Handle button4, clean up data
        if button2:
            with st.spinner("Cleaning up data..."):
                try:
                    self.backend.cleanup_data()

                    st.success("Cleanup complete!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error during cleanup: {e}")

        # Handle button click and prompt submission
        if button3 and prompt:
            st.session_state['processing'] = True

            with st.spinner("Processing..."):
                try:
                    generated_response = self._generate_llm_response(prompt, chat_history=st.session_state['chat_history'])

                    st.session_state['prompt_history'].append(prompt)
                    st.session_state['answer_history'].append(generated_response)

                    st.session_state['chat_history'].append(('human', prompt))
                    st.session_state['chat_history'].append(('ai', generated_response['result']))

                    st.success("Done!")
                    st.balloons()

                except Exception as e:
                    st.error(f"Error: {e}")

                finally:
                    st.session_state['processing'] = False

        # Display chat messages from history on app rerun
        if st.session_state['answer_history']:
            for generated_response, prompt in zip(st.session_state['answer_history'], st.session_state['prompt_history']):
                st.chat_message("user").write(prompt)
                st.chat_message("assistant").write(generated_response['result'])


    # What is LangChain?
    # What did i just ask you?
