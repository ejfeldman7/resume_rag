from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def generate_response(self, passage, query):
        inputs = self.tokenizer([passage + " " + query], return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
