from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Chatbot:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer()
        self.corpus_embeddings = self.vectorizer.fit_transform(corpus)
        self.corpus = corpus

    def retrieve_passages(self, query):
        query_embedding = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        best_idx = similarity_scores.argmax()
        return self.corpus[best_idx]

    def generate_response(self, passage, query):
        return f"Passage: {passage}\nQuery: {query}\n"
