import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from streamlit_mic_recorder import mic_recorder
from utils import get_timestamp, load_config, get_avatar
from image_handler import handle_image
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from database_operations import load_last_k_text_messages, get_mean_age, save_text_message, save_image_message, save_audio_message, load_messages, get_all_chat_history_ids, delete_chat_history
import sqlite3
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline   
import re
import numpy as np
import enchant
from language_tool_python import LanguageTool

config = load_config()
enchant_dict = enchant.Dict("en_US")

@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain()
    return load_normal_chain()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def predict_age(user_input):
    model_filename = 'age_predictor_model.pkl'
    model = joblib.load(model_filename)
    predicted_age = model.predict([user_input])
    return predicted_age[0]

def spell_check(text):
    words = text.split()
    misspelled_words = [word for word in words if not enchant_dict.check(word)]
    return misspelled_words

# def grammar_check(text):
#     try:
#         tool = LanguageTool('en-US')
#         matches = tool.check(text)
#         return matches
#     except Exception as e:
#         st.error(f"Error: {e}")
#         return []

def calculate_errors(messages):
    user_messages = [message["content"] for message in messages if message["sender_type"] == "human"]
    spelling_errors = [spell_check(message) for message in user_messages]
    return spelling_errors

def analyze_proficiency(spelling_errors, messages):
    user_messages = [message["content"] for message in messages if message["sender_type"] == "human"]
    total_errors = sum(len(errors) for errors in spelling_errors)
    proficiency_score = 1 - (total_errors / sum(len(message.split()) for message in user_messages))  # Assuming higher score implies better proficiency
    return proficiency_score

def main():
    st.title("CHAT BOT")
    st.write(css, unsafe_allow_html=True)

    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
        st.session_state.predicted_age = None
    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    with voice_rec_col:
        voice_recording = mic_recorder(start_prompt="Record Audio", stop_prompt="Stop recording", just_once=True)
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)

    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")

    uploaded_audio = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"], key=st.session_state.audio_uploader_key)
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    uploaded_pdf = st.sidebar.file_uploader("Upload a pdf file", accept_multiple_files=True, 
                                            key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat)

    if uploaded_pdf:
        with st.spinner("Processing pdf..."):
            add_documents_to_db(uploaded_pdf)
            st.session_state.pdf_uploader_key += 2

    if uploaded_audio:
        transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input="Summarize this text: " + transcribed_audio, chat_history=[])
        save_audio_message(get_session_key(), "human", uploaded_audio.getvalue())
        save_text_message(get_session_key(), "ai", llm_answer)
        st.session_state.audio_uploader_key += 2

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording["bytes"])
        print(transcribed_audio)
        llm_chain = load_chain()
        llm_answer = llm_chain.run(user_input=transcribed_audio, 
                                   chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
        save_audio_message(get_session_key(), "human", voice_recording["bytes"])
        save_text_message(get_session_key(), "ai", llm_answer)

    if user_input:
        if uploaded_image:
            predicted_age = predict_age(user_input)
            with st.spinner("Processing image..."):
                llm_answer = handle_image(uploaded_image.getvalue(), user_input)
                save_text_message(get_session_key(), "human", user_input, predicted_age)
                save_image_message(get_session_key(), "human", uploaded_image.getvalue(), predicted_age)
                save_text_message(get_session_key(), "ai", llm_answer, predicted_age)
                user_input = None
        if user_input:
            predicted_age = predict_age(user_input)
            llm_chain = load_chain()
            llm_answer = llm_chain.run(user_input=user_input, 
                                       chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"]))
            save_text_message(get_session_key(), "human", user_input, predicted_age)
            save_text_message(get_session_key(), "ai", llm_answer, predicted_age)
            st.session_state.predicted_age = predicted_age
            user_input = None

    # Add Predict Age button
    if st.session_state.predicted_age is not None:
        predict_age_button_label = f"Predicted Age: {get_mean_age(st.session_state.session_key)}"
    else:
        predict_age_button_label = "Predict Age"

    st.button(predict_age_button_label, key="predict_age_button", disabled=True)

    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio": 
                        st.audio(message["content"], format="audio/wav")

        # Perform error calculation and proficiency analysis here
        spelling_errors = calculate_errors(chat_history_messages)
        proficiency_score = analyze_proficiency(spelling_errors, chat_history_messages)
        if proficiency_score < 0.4:
            proficiency_type = "Beginner"
        if proficiency_score >= 0.4 and proficiency_score <= 0.8:
            proficiency_type = "Intermediate"
        if proficiency_score > 0.8:
            proficiency_type = "Advanced"

        if st.button("Show Error and Proficiency Analysis Results"):
            st.write("Spelling Errors:", sum(len(errors) for errors in spelling_errors))
            st.write("Proficiency Score:", proficiency_score)
            st.write("Control over Language:", proficiency_type)
        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()
