import os
import hashlib

from PyPDF2 import PdfReader
import streamlit as st

from utils import vector_search
from utils.chatbot import ResumeChatBot


def extract_text_from_pdf(pdf_file: str):
    '''
    Extracts text from a PDF file
    Args:
        pdf_file (str): Path to the PDF file
    Returns:
        str: Extracted text from the PDF file
    '''
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text: str, chunk_size: int=100):
    '''
    Converts input text string into chunks of specified size to be used for vectorization
    Args:
        text (str): Input text string
        chunk_size (int): Size of each chunk
    Returns:
        list: List of text chunks
    '''
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


@st.cache_data
def load_preexisting_pdfs(chatbot: ResumeChatBot):
    '''
    Loads all pre-existing PDFs and creates FAISS index for each PDF
    Args:
        chatbot (ResumeChatBot): Chatbot object
    Returns:
        dict: Dictionary containing FAISS index for each PDF
    '''
    pdf_data = {}
    for filename in os.listdir('data'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join('data', filename)
            pdf_data[filename] = vector_search.load_or_create_faiss_index(pdf_path, chatbot)
    return pdf_data


def get_pdf_hash(pdf_content: bytes):
    '''
    Generates MD5 hash for the input PDF content
    Args:
        pdf_content (bytes): PDF content in bytes
    Returns:
        str: MD5 hash of the PDF content
    '''
    return hashlib.md5(pdf_content).hexdigest()
