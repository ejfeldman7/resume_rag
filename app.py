import streamlit as st
from utils.pdf_extract import PDFExtractor
from utils.embeddings import EmbeddingRetriever
from utils.chatbot import Chatbot


pdf_path = "data/feldman_resume.pdf"
resume_text = PDFExtractor.extract_text_from_pdf(pdf_path)

retriever = EmbeddingRetriever()
resume_embedding = retriever.get_embedding(resume_text)

corpus = [resume_text]
chatbot = Chatbot(corpus)

# Streamlit app
st.title("Resume Chatbot")
st.write("Ask me anything about my resume!")

query = st.text_input("Enter your question:")
if query:
    passage = retriever.retrieve_passages(query, resume_text, resume_embedding)
    response = chatbot.generate_response(passage, query)
    st.write(response)
