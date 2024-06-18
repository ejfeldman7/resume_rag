import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration  # noqa

from utils.embeddings import EmbeddingRetriever
from utils.pdf_extract import PDFExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chatbot:
    def __init__(self, pdf_path, logger=None):
        self.pdf_text = PDFExtractor.extract_text_from_pdf(pdf_path)
        self.embedding_retriever = EmbeddingRetriever()
        self.vectorizer = TfidfVectorizer()
        self.corpus_embeddings = self.vectorizer.fit_transform([self.pdf_text])
        self.chunk_size = 1000  # Number of characters per chunk
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.logger = logger if logger else logging.getLogger(__name__)

    def chunk_document(self):
        chunks = []
        start_idx = 0
        while start_idx < len(self.pdf_text):
            end_idx = start_idx + self.chunk_size
            if end_idx >= len(self.pdf_text):
                end_idx = len(self.pdf_text)
            chunks.append(self.pdf_text[start_idx:end_idx])
            start_idx = end_idx
        return chunks

    def retrieve_passages(self, query):
        query_embedding = self.embedding_retriever.get_embedding(query)
        logger.info(f"QE Shape: {query_embedding.shape}")
        logger.info(f"CE shape: {self.corpus_embeddings.shape}")
        if query_embedding.shape[1] > self.corpus_embeddings.shape[1]:
            zeros_padding = np.zeros((query_embedding.shape[0], self.corpus_embeddings.shape[1] - query_embedding.shape[1]))
            query_embedding = np.hstack((query_embedding, zeros_padding))
        if query_embedding.shape[1] < self.corpus_embeddings.shape[1]:
            zeros_padding = np.zeros((self.corpus_embeddings.shape[1], query_embedding.shape[0] - self.corpus_embeddings.shape[1]))
            query_embedding = np.hstack((self.corpus_embeddings.shape[1], zeros_padding))
        logger.info(f"QE Shape: {query_embedding.shape}")
        logger.info(f"CE shape: {self.corpus_embeddings.shape}")
        scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        best_idx = scores.argmax()
        relevant_chunk = self.chunk_document()[best_idx]
        logger.info(f"Query: {query}, Relevant Chunk: {relevant_chunk}")
        return relevant_chunk

    def generate_response(self, passage, query):
        input_text = f"summarize: {passage}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(input_ids)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info(f"Passage: {passage}, Query: {query}, Summary: {summary}")
        return f"Summary: {summary}\nQuery: {query}"
