import sys
import os
import streamlit as st

# Ensure Streamlit can find chatbot.py
sys.path.append(os.path.dirname(__file__))

# Import functions from chatbot.py
from chatbot import generate_response, save_knowledge, search_knowledge

# Streamlit Page Configuration
st.set_page_config(page_title="AI Chatbot", layout="wide")

# Title and Layout
st.title("AI Chatbot with Knowledge Base")
st.sidebar.header("Upload Knowledge")

# Chat Section
st.subheader("Chat with the AI")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# User input and response generation
user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input:
        st.session_state.history.append(f"You: {user_input}")

        # First search in the knowledge base
        search_result = search_knowledge(user_input)
        
        # If no relevant knowledge found, generate response using GPT-J
        response = search_result if search_result else generate_response(user_input)

        st.session_state.history.append(f"AI: {response}")

# Display chat history
for message in st.session_state.history:
    st.write(message)

# Knowledge Upload Section
st.sidebar.subheader("Upload Documents")

uploaded_file = st.sidebar.file_uploader("Upload a text file", type=['txt', 'pdf'])

if uploaded_file:
    # Handle text or PDF file upload
    content = ""
    if uploaded_file.type == "application/pdf":
        # Handle PDF file
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            content += page.extract_text()
    else:
        # Handle text file
        content = uploaded_file.read().decode('utf-8')

    # Input for knowledge title and save button
    title = st.sidebar.text_input("Title for this knowledge:")
    if st.sidebar.button("Save Knowledge") and title:
        save_knowledge(title, content)
        st.sidebar.success("Knowledge saved successfully!")