import logging
from typing import Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import streamlit as st
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeChatBot:
    @staticmethod
    @st.cache_resource
    def load_encoder(_model: str) -> Any:
        return SentenceTransformer(_model)
 
    @staticmethod
    @st.cache_resource
    def load_generator(_model: str) -> Any:
        return AutoModelForSeq2SeqLM.from_pretrained(_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    @staticmethod
    @st.cache_resource
    def load_tokenizer(_model: str) -> Any:
        return AutoTokenizer.from_pretrained(_model)
    
    def __init__(self, encoder: str = "paraphrase-MiniLM-L6-v2", generator: str = "google/flan-t5-small"):
        try:
            self.encoder = self.load_encoder(encoder)
            logger.info(f"Encoder for ({encoder}) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load encoder ({encoder}): {str(e)}")
            raise

        try:
            self.generator = self.load_generator(generator)
            logger.info(f"Generator for ({generator}) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load generator ({generator}): {str(e)}")
            raise

        try:
            self.tokenizer = self.load_tokenizer(generator)
            logger.info(f"Tokenizer for ({generator}) loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for ({generator}): {str(e)}")
            raise

        self.generator.eval()

    def extract_key_facts(self, text: str):
        '''
        Extracts key facts from the input text
        Args:
            text (str): Input text string
        Returns:
            list: List of key facts extracted from the input text
        '''
        # TODO: NER or other NLP techniques for better extraction.
        return [phrase.strip() for phrase in text.split(',')]

    def get_embedding(self, text: str):
        '''
        Generates embeddings for the input text string
        Args:
            text (str): Input text string
        Returns:
            numpy.ndarray: Embedding for the input text
        '''
        logger.info(f"Getting embedding for text: {text[:10]}...")
        return self.encoder.encode(text)

    def select_relevant_chunks(self, query: str, chunks: list, top_n: int = 3):
        '''
        Selects the top N relevant chunks based on the similarity with the input query
        Args:
            query (str): Input query string
            chunks (list): List of chunk strings
            top_n (int): Number of top chunks to select
        Returns:    
            list: List of top N relevant chunks
        '''
        query_embedding = self.get_embedding(query)
        chunk_embeddings = [self.get_embedding(chunk) for chunk in chunks]

        similarities = [np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)) 
                        for chunk_emb in chunk_embeddings]
        
        top_indices = np.argsort(similarities)[-top_n:]
        return [chunks[i] for i in top_indices]

    @torch.no_grad()
    def get_response(self, context: str, question: str):
        '''
        Generates response for the input question given the context of the conversation
        Args:
            context (str): Context of the conversation
            question (str): Question asked by the user
        Returns:
            str: Response to the from the chatbot
        '''
        logger.info(f"Getting response for question: {question[:10]}...")
        prompt = f"""Based on the following resume information, answer the question accurately and in detail. If the information is not explicitly mentioned in the resume, say "I don't have enough information to answer that."

Resume information:
{context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Simple fact-checking
        key_facts = self.extract_key_facts(response)
        for fact in key_facts:
            if fact.lower() not in context.lower():
                response += "\n\nNote: I'm not entirely certain about some details in this response. Please verify against the original resume."
                break

        return response


@st.cache_resource
def initialize_chatbot():
    return ResumeChatBot()
