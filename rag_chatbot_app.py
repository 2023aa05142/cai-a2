import streamlit as st
from rag_chatbot import ragChatbot

st.set_page_config(page_title="CAI (S1-24) | Assignment 2 | Group 51")

st.markdown(
    """
    <style>
        .st-emotion-cache-1c7y2kd {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True)

#st.title("Conversational AI (S1-24) | Assignment 2 | Group 51")
#st.title("CAI (S1-24) | Assignment 2 | Group 51")
#st.header("Chatbot by Group 51")
st.subheader("Group 51's Chatbot")

chatbot = ragChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = chatbot.answer("Test")
        score = str(response.score)
        message = response.answer+'<br/><sup style="font-size:0.7em">Confidence: '+score+'%</sup>'
        message_placeholder.markdown(message, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": message})