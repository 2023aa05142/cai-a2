import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from document_store import DocumentStore
from basic_chatbot import BasicChatbot
from advance_chatbot import AdvanceChatbot
from guard_rail import GuardRail

# Load a Small Open-Source Language Model
@st.cache_resource
def load_language_model(model_name: str = "google/flan-t5-base"):
    print("::load_language_model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def getChatbotResources(index):
    print("::getChatbotResources")
    # Load the language model
    tokenizer, model = load_language_model()
    store = DocumentStore()
    chatbots = [BasicChatbot(store, tokenizer, model), AdvanceChatbot(store, tokenizer, model)]
    guardrail = GuardRail()
    return store, guardrail, chatbots[index]

st.set_page_config(page_title="CAI (S1-24) | Group 51 | Assignment 2")

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

st.subheader("Group 51's Chatbot")

st.sidebar.header("Chatbot settings")
with st.sidebar:
    techniques = ["Basic", "Advance"]
    selected_technique = st.selectbox("Chatbot's response generation technique:", techniques, 1)
    selected_technique_index = techniques.index(selected_technique)
    # threshold for relevance
    threshold = st.number_input("Relevancy threshold for multi stage retrievals", 0.5, format="%0.1f")
    uploaded_files = st.sidebar.file_uploader("Documents required for RAG technique", type="pdf", accept_multiple_files=True)

# Update session state to handle multiple selected models
if 'selected_technique' not in st.session_state or st.session_state.selected_technique != selected_technique:
    st.session_state.selected_technique = selected_technique
    #st.write(f"Selected technique: {st.session_state.selected_technique}")
if 'selected_technique_index' not in st.session_state or st.session_state.selected_technique_index != selected_technique_index:
    st.session_state.selected_technique_index = selected_technique_index
    #st.write(f"Selected technique index: {st.session_state.selected_technique_index}")

store, guardrail, chatbot = getChatbotResources(selected_technique_index)

if uploaded_files:
    isDocumentAdded = False
    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            isDocumentAdded = store.addDocument(uploaded_file.name, uploaded_file.read()) or isDocumentAdded

    print("Documents processed.")
    st.sidebar.success("Documents processed.")

    if(isDocumentAdded):
        print("Generating embeddings...")
        with st.spinner("Generating embeddings..."):
            store.buildEmbeddings()

    st.sidebar.success("Let's chat!")

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
        message = "This financial question is not allowed being either irrelevant or harmful."
        message_placeholder = st.empty()
        result, _ = guardrail.validate_input(prompt)
        if (result):
            response = chatbot.answer(prompt, threshold)
            answer = response.answer
            confidence = "{:.2f}".format(response.confidence*100)
            print(f"Answer: {answer}\nConfidence: {confidence}")
            result, _ = guardrail.validate_response(answer)
            message = "The response of this question is not allowed as it contains either hallucinated or misleading information."
            if (result):
                message = answer+'<br/><sup style="font-size:0.7em">Confidence: '+confidence+'%</sup>'
        message_placeholder.markdown(message, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": message})