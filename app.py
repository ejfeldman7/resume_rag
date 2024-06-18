import streamlit as st
from resume_rag.utils.pdf_extract import PDFExtractor
from resume_rag.utils.embeddings import EmbeddingRetriever
from resume_rag.utils.chatbot import Chatbot


pdf_path = "data/feldman_resume.pdf"
resume_text = PDFExtractor.extract_text_from_pdf(pdf_path)

retriever = EmbeddingRetriever()
resume_embedding = retriever.get_embedding(resume_text)

chatbot = Chatbot()

# Streamlit app
st.title("Resume Chatbot")
st.write("Ask me anything about my resume!")

query = st.text_input("Enter your question:")
if query:
    passage = retriever.retrieve_passages(query, resume_text, resume_embedding)
    response = chatbot.generate_response(passage, query)
    st.write(response)
