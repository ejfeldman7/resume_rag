from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

class Chatbot:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
        self.generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

    def generate_response(self, query, passage):
        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.generator(**inputs, return_dict_in_generate=True)
        return self.tokenizer.decode(outputs[0]["output_ids"], skip_special_tokens=True)

