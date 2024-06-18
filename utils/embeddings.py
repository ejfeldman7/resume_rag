from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer("distilbert-base-nli-stsb-mean-token")

    def get_embedding(self, text):
        return self.embedder.encode([text])

    def retrieve_passages(self, query, text, text_embedding):
        query_embedding = self.embedder.encode([query])
        scores = cosine_similarity(query_embedding, text_embedding)[0]
        best_idx = scores.argmax()
        return text[best_idx]
