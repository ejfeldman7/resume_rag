from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from embeddings import EmbeddingRetriever
from pdf_extract import PDFExtractor


class Chatbot:
    def __init__(self, pdf_path):
        self.pdf_text = PDFExtractor.extract_text_from_pdf(pdf_path)
        self.embedding_retriever = EmbeddingRetriever()
        self.vectorizer = TfidfVectorizer()
        self.corpus_embeddings = self.vectorizer.fit_transform([self.pdf_text])
        self.chunk_size = 1000  # Number of characters per chunk
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")

    def chunk_document(self):
        chunks = []
        for i in range(0, len(self.pdf_text), self.chunk_size):
            chunk = self.pdf_text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def retrieve_passages(self, query):
        query_embedding = self.embedding_retriever.get_embedding(query)
        scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        best_idx = scores.argmax()
        relevant_chunk = self.chunk_document()[best_idx]
        return relevant_chunk

    def generate_response(self, passage, query):
        input_text = f"summarize: {passage}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(input_ids)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return f"Summary: {summary}\nQuery: {query}"
