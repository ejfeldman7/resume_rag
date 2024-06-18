from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("distilgpt2")

    def generate_response(self, passage, query):
        inputs = self.tokenizer([passage + " " + query], return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
