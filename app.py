import streamlit as st

from utils.chatbot import ResumeChatBot
from utils import pdf_helpers
from utils import vector_search


st.title("Resume-based Chatbot")

chatbot = ResumeChatBot()

preexisting_pdfs = pdf_helpers.load_preexisting_pdfs(chatbot)

pdf_option = st.radio("Choose a PDF option:", ("Pre-existing PDF", "Upload your own"))

if pdf_option == "Pre-existing PDF":
    selected_pdf = st.selectbox("Select a pre-existing PDF:", list(preexisting_pdfs.keys()))
    current_pdf_data = preexisting_pdfs[selected_pdf]
else:
    uploaded_file = st.file_uploader("Upload a PDF resume", type="pdf")
    if uploaded_file is not None:
        current_pdf_data = vector_search.load_or_create_faiss_index(uploaded_file, chatbot, is_upload=True)
        st.success("Resume uploaded and processed successfully!")

if 'current_pdf_data' in locals():
    user_query = st.text_input("Ask a question about the resume:")
    if user_query:
        relevant_chunk = vector_search.find_most_similar_chunk_faiss(user_query, current_pdf_data['index'], current_pdf_data['chunks'], chatbot)
        response = chatbot.get_response(relevant_chunk, user_query)
        st.write("Chatbot:", response)
