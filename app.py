import sys
import os
sys.path.append(os.path.dirname(__file__))
from chatbot import generate_response, save_knowledge, search_knowledge

import streamlit as st
from chatbot import generate_response, save_knowledge, search_knowledge

st.set_page_config(page_title="AI Chatbot", layout="wide")

# Title and Layout
st.title("AI Chatbot with Knowledge Base")
st.sidebar.header("Upload Knowledge")

# Chat Section
st.subheader("Chat with the AI")

if 'history' not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.history.append(f"You: {user_input}")
        response = search_knowledge(user_input) or generate_response(user_input)
        st.session_state.history.append(f"AI: {response}")

for message in st.session_state.history:
    st.write(message)

# Knowledge Upload Section
st.sidebar.subheader("Upload Documents")

uploaded_file = st.sidebar.file_uploader("Upload a text file", type=['txt', 'pdf'])

if uploaded_file:
    content = uploaded_file.read().decode('utf-8')
    title = st.sidebar.text_input("Title for this knowledge:")
    if st.sidebar.button("Save Knowledge"):
        save_knowledge(title, content)
        st.sidebar.success("Knowledge saved successfully!")