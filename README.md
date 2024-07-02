# resume_rag
This repository contains the code and underlying resume data to run a [Streamlit webapp](https://resume-rag-bot.streamlit.app/) for a RAG chatbot.

## Contents
## app.py
This is the main application that runs the [Streamlit webapp](https://resume-rag-bot.streamlit.app/)
## Utils
Utils contains three underlying modules that are imported and help run app.py
- chatbot.py
- pdf_helpers.py
- vector_search.py

## Details
### chatbot.py
This module defines the `ResumeChatBot` class, which is resposible for loading our models and generating responses.
This class loads Google's `flan-t5-base` model currently for both tokenization, in the `get_embedding` method, and generation, in the `generate_response` method. 

### pdf_helpers.py
This module defines a variety of functions that are resposible for loading PDF files, parsing their text to strings, and breaking them into chunks.
These functions handle either preexisting PDFs, which can be found in the `/data` directory, or those uploaded by users

### vector_search.py
This module uses `FAISS` to enable a VectorDB to search over the chunks from our parsed resume when a user supplies an input query.
Functions here include loading or creating our FAISS indices and finding our most similar chunks.

### /data
This directory contains the default PDF resume file to be used for for augmented generation.

## Future Work
Currently, responses from the chatbot are generally on topic, but leave room for improvement. Much of the limiting factors come to memory constraints and trying to balance
having a smaller model with generating meaningful results. This could ideally be alleviated by deploying the model in a separate location or using an existing API for a larger
model that is more powerful. Those implementations may not be free, so they were not pursued in this initial approach. 
