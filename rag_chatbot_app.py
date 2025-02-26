import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from document_store import DocumentStore
from basic_chatbot import BasicChatbot
from advance_chatbot import AdvanceChatbot
from guard_rail import GuardRail

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

def validate_query_with_model(query: str, validation_model_name: str = "facebook/bart-large-mnli") -> bool:
    classifier = pipeline("zero-shot-classification", model=validation_model_name)
    candidate_labels = ["financial question", "irrelevant", "harmful"]
    result = classifier(query, candidate_labels)
    top_label = result["labels"][0]
    return top_label == "financial question"

def filter_response(response: str, filtering_model_name: str = "facebook/bart-large-mnli") -> bool:
    """
    Filters generated responses using a zero-shot classification model to ensure relevance and correctness.

    :param response: Generated response to validate.
    :param filtering_model_name: Pre-trained model for classification.
    :return: True if the response is valid, False otherwise.
    """
    classifier = pipeline("zero-shot-classification", model=filtering_model_name)
    candidate_labels = ["relevant", "hallucinated", "misleading"]
    result = classifier(response, candidate_labels)
    top_label = result["labels"][0]
    return top_label == "relevant"

# Load a Small Open-Source Language Model
@st.cache_resource
def load_language_model(model_name: str = "google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #tokenizer, model = None, None
    return tokenizer, model

@st.cache_resource
def getChatbot(index):
    # Load the language model
    tokenizer, model = load_language_model()
    store = DocumentStore()
    chatbots = [BasicChatbot(store, tokenizer, model), AdvanceChatbot(store, tokenizer, model)]
    guardrail = GuardRail()
    return store, guardrail, chatbots[index]
    #return store, None

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

#st.title("Conversational AI (S1-24) | Assignment 2 | Group 51")
#st.title("CAI (S1-24) | Assignment 2 | Group 51")
#st.header("Chatbot by Group 51")
st.subheader("Group 51's Chatbot")

st.sidebar.header("Chatbot settings")
with st.sidebar:
    techniques = ["Basic", "Advance"]
    selected_technique = st.selectbox("Chatbot's response generation technique:", techniques, 0)
    selected_technique_index = techniques.index(selected_technique)
    uploaded_files = st.sidebar.file_uploader("Documents required for RAG technique", type="pdf", accept_multiple_files=True)

# Update session state to handle multiple selected models
if 'selected_technique' not in st.session_state or st.session_state.selected_technique != selected_technique:
    st.session_state.selected_technique = selected_technique
    #st.write(f"Selected technique: {st.session_state.selected_technique}")
if 'selected_technique_index' not in st.session_state or st.session_state.selected_technique_index != selected_technique_index:
    st.session_state.selected_technique_index = selected_technique_index
    #st.write(f"Selected technique index: {st.session_state.selected_technique_index}")

store, guardrail, chatbot = getChatbot(selected_technique_index)

if uploaded_files:
    isDocumentAdded = False
    with st.spinner("Processing documents..."):
        # input_dir = "./financial-statements/raw_documents"
        # output_dir = "./financial-statements/cleaned_documents"
        # os.makedirs(input_dir, exist_ok=True)
        # os.makedirs(output_dir, exist_ok=True)

        for uploaded_file in uploaded_files:
            #file_path = os.path.join(input_dir, uploaded_file.name)
            isDocumentAdded = store.addDocument(uploaded_file.name, uploaded_file.read()) or isDocumentAdded
            #with open(file_path, "wb") as f:
                #f.write(uploaded_file.read())

    print("Documents processed.")
    st.sidebar.success("Documents processed.")

    if(isDocumentAdded):
        print("Generating embeddings...")
        with st.spinner("Generating embeddings..."):
            store.buildEmbeddings()

    st.sidebar.success("Let's chat!")

# st.sidebar.header("Upload documents")
# uploaded_files = st.sidebar.file_uploader("Upload financial documents (PDF format)", type="pdf", accept_multiple_files=True)
# if uploaded_files:
#     with st.spinner("Processing docuemnts..."):
#         input_dir = "./financial-statements/pdfs"
#         output_dir = "./financial-statements/cleaned_texts"
#         os.makedirs(input_dir, exist_ok=True)
#         os.makedirs(output_dir, exist_ok=True)

#         for uploaded_file in uploaded_files:
#             file_path = os.path.join(input_dir, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.read())

#         results = preprocess_pdfs(input_dir, output_dir)

#     st.success("Documents processed successfully!")

#     all_chunks = []
#     for result in results:
#         with open(result["cleaned_file"], "r", encoding="utf-8") as f:
#             text = f.read()
#             chunks = split_into_chunks(text)
#             all_chunks.extend(chunks)

#     with st.spinner("Generating embeddings..."):
#         embeddings = embed_text_chunks(all_chunks)
#         vector_db = create_vector_database(embeddings)

#     st.sidebar.success("Ready to accept queries!")

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
            response = chatbot.answer(prompt)
            answer = response.answer
            confidence = "{:.2f}".format(response.confidence*100)

            print(answer)
            print(confidence)
            for i, chunk in enumerate(response.chunks[:5]):
                print(f"**Result {i+1}:**")
                print(chunk)

            result, _ = guardrail.validate_response(answer)
            message = "The response of this question is not allowed as it contains either hallucinated or misleading information."
            if (result):
                message = answer+'<br/><sup style="font-size:0.7em">Confidence: '+confidence+'%</sup>'
        message_placeholder.markdown(message, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": message})