import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeChatBot:
    def __init__(self, model_name:str="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer for ({model_name}) loaded successfully")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info(f"Model ({model_name}) loaded successfully")

    def get_embedding(self, text:str):
        '''
        Generates embeddings for the input text string
        Args:
            text (str): Input text string
        Returns:
            numpy.ndarray: Embedding for the input text
        '''
        logger.info(f"Getting embedding for text: {text[:10]}...")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logger.info("Generating torch embeddings for text")
            outputs = self.model.encoder(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def get_response(self, context:str, question:str):
        '''
        Generates response for the input question given the context of the conversation
        Args:
            context (str): Context of the conversation
            question (str): Question asked by the user
        Returns:
            str: Response to the from the chatbot
        '''
        logger.info(f"Getting response for question: {question[:10]}...")
        input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
