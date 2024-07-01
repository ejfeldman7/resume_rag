import os

import numpy as np
import pickle
import streamlit as st
import faiss

from utils import pdf_helpers
from utils.chatbot import ResumeChatBot


@st.cache_data
def load_or_create_faiss_index(pdf_path: str, _chatbot: ResumeChatBot, is_upload: bool=False):
    '''
    Loads or creates FAISS index for the input PDF file
    Args:
        pdf_path (str): Path to the PDF file
        chatbot (ResumeChatBot): Chatbot object
        is_upload (bool): Flag to indicate if the PDF is uploaded by the user
    Returns:
        dict: Dictionary containing FAISS index and text chunks for the PDF
    '''
    pdf_content = open(pdf_path, 'rb').read() if not is_upload else pdf_path.read()
    pdf_hash = pdf_helpers.get_pdf_hash(pdf_content)
    index_path = f"faiss_index_{pdf_hash}.pkl"

    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    else:
        pdf_text = pdf_helpers.extract_text_from_pdf(pdf_content)
        chunks = pdf_helpers.chunk_text(pdf_text)
        embeddings = np.array([_chatbot.get_embedding(chunk) for chunk in chunks])
        faiss_index = create_faiss_index(embeddings)

        data = {
            'index': faiss_index,
            'chunks': chunks
        }

        with open(index_path, 'wb') as f:
            pickle.dump(data, f)

        return data


def find_most_similar_chunk_faiss(query: str, index: faiss.IndexFlatL2, chunks: str, _chatbot: ResumeChatBot):
    '''
    Finds the most similar chunk to the input query using FAISS index
    Args:
        query (str): Input query
        index (faiss.IndexFlatL2): FAISS index for the chunks
        chunks (str): List of text chunks
        chatbot (ResumeChatBot): Chatbot object
    Returns:
        str: Most similar chunk to the input query
    '''
    query_embedding = _chatbot.get_embedding(query).reshape(1, -1)
    _, indices = index.search(query_embedding, 1)
    return chunks[indices[0][0]]


def create_faiss_index(embeddings: np.ndarray):
    '''
    Creates a FAISS index for the input embeddings
    Args:
        embeddings (numpy.ndarray): Input embeddings
    Returns:
        faiss.IndexFlatL2: FAISS index for the embeddings
    '''
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
