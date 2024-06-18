from transformers import AutoModelForCausalLM, AutoTokenizer

class Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def generate_response(self, passage, query):
        input_text = passage + " " + query
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        output = self.model.generate(input_ids, max_length=150, num_return_sequences=3, do_sample=True)
        responses = [self.tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output]

        return responses
