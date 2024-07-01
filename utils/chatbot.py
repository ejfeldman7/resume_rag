import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeChatBot:
    def __init__(self, model_name:str="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Tokenizer for ({model_name}) loaded successfully")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logger.info(f"Model ({model_name}) loaded successfully")

    def extract_key_facts(self, text: str):
        '''
        Extracts key facts from the input text
        Args:
            text (str): Input text string
        Returns:
            list: List of key facts extracted from the input text
        '''
        #TODO: NER or other NLP techniques for better extraction.
        return [phrase.strip() for phrase in text.split(',')]

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
        prompt = f"""Based on the following resume information, answer the question accurately and in detail. If the information is not explicitly mentioned in the resume, say "I don't have enough information to answer that."

Resume information:
{context}

Question: {question}

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Simple fact-checking
        key_facts = self.extract_key_facts(response)
        for fact in key_facts:
            if fact.lower() not in context.lower():
                response += "\n\nNote: I'm not entirely certain about some details in this response. Please verify against the original resume."
                break

        return response
