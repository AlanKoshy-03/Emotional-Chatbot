# README
# Emotional Chatbot
## Overview

The Emotional Chatbot is a Streamlit-based application that leverages large language models (LLMs) for interactive text, image, and audio processing. The application supports various functionalities including chat sessions, voice recording, file uploads (images, audio, and PDFs), age prediction from text input, spelling and grammar checks, and proficiency analysis.

## Features
***Interactive Chat:*** Engage in text-based conversations with AI.

***Voice Recording and Transcription:*** Record audio or upload audio files for transcription and processing.

***Image Handling:*** Upload images and process them with text input.

***PDF Chat:*** Upload PDF files for content extraction and chat-based interactions.

***Age Prediction:*** Predict user's age based on text input.

***Spelling and Grammar Checks:*** Analyze and correct spelling and grammatical errors.

***Proficiency Analysis:*** Calculate and display proficiency scores based on user inputs.



## Installation
**1. Clone the repository:**
```bash
git clone https://github.com/AlanKoshy-03/Emotional-Chatbot.git
cd Emotional-Chatbot
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
Ensure you have the necessary models for age prediction and LLMs. This may include downloading pre-trained models and placing them in the correct directory as shown below.

Download the models folder from the following link

https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main

https://huggingface.co/openai/whisper-small

https://huggingface.co/BAAI/bge-large-en-v1.5

## Configuration
A configuration file ***config.yaml*** is used to manage settings such as database paths and chat configurations. Ensure this file is correctly set up before running the app.

## Usage
**1. Run the Streamlit app:**
```bash
python database_operations.py

```
```
streamlit run app.py

```

**2. Interact with the app:**

Use the sidebar to upload files, toggle PDF chat, and manage chat sessions.
Enter text input directly in the chat interface or use the voice recorder to transcribe and process audio.
View the chat history and analyze proficiency scores based on your interactions.

## File Structure
***app.py:*** Main application file that handles the Streamlit interface and core functionalities.

***llm_chains.py:*** Contains functions to load different LLM chains.

***models:*** Contains all the required models along with the llava folder as shown below:
```bash
|-- models

     |-- mistral-7b-instruct-v0.1.Q3_K_M.gguf
     
     |-- mistral-7b-instruct-v0.1.Q5_K_M.gguf
     
     |-- llava
     
           |-- llava_ggml-model-q5_k.gguf
           
           |-- mmproj-model-f16.gguf
 ```          
***streamlit_mic_recorder.py:*** Custom component for recording audio in Streamlit.

***utils.py:*** Utility functions for the application.

***image_handler.py:*** Functions for handling image uploads and processing.

***audio_handler.py:*** Functions for audio transcription and processing.

***pdf_handler.py:*** Functions for adding and managing PDF documents.

***database_operations.py:*** Functions for interacting with the SQLite database.

***requirements.txt:*** List of dependencies required for the project.

***config.yaml:*** Configuration file for managing settings.

## Acknowledgements
***Streamlit*** for providing the interactive app framework.

***language_tool_python*** for finding spelling errors.

***sklearn*** for machine learning model support.

*All other libraries and tools used in this project.*

## Contact
*For any questions or feedback, please contact:*

aalbin.joseph@btech.christuniversity.in

alan.koshy@btech.christuniversity.in

mervyn.varghese@btech.christuniversity.in
