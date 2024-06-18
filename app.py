import streamlit as st
from utils.pdf_extract import PDFExtractor
from utils.chatbot import Chatbot


pdf_path = "data/feldman_resume.pdf"
chatbot = Chatbot(pdf_path)

# Streamlit app
st.title("Resume Chatbot")
st.write("Ask me anything about my resume!")

query = st.text_input("Enter your question:")
if query:
    passage = chatbot.retrieve_passages(query)
    response = chatbot.generate_response(passage, query)
    st.write(response)
