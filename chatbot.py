import os
import json
import torch
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Copy the secrets to a new dictionary for modification
firebase_key = dict(st.secrets["firebase"])

# Fix special characters in the private key
firebase_key["private_key"] = firebase_key["private_key"].replace("\\n", "\n")

# Save the credentials as a temporary JSON file
with open("temp_firebase_key.json", "w") as json_file:
    json.dump(firebase_key, json_file, indent=4)

# Debug: Print the contents of the temporary JSON file
with open("temp_firebase_key.json", "r") as json_file:
    print("Temporary Firebase Key JSON:", json_file.read())

# Use the temporary file to initialize Firebase
cred = credentials.Certificate("temp_firebase_key.json")
if not firebase_admin._apps:  # Prevents duplicate initialization
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Remove the temporary file after initialization
os.remove("temp_firebase_key.json")

# Initialize GPT-J-6B Model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cpu")  # Use CPU for deployment compatibility

# Initialize Sentence Transformer for Knowledge Search
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate a response using GPT-J-6B
def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, top_k=60)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to save knowledge to Firebase
def save_knowledge(title, content):
    doc_ref = db.collection("knowledge_base").document(title)
    doc_ref.set({
        "title": title,
        "content": content
    })

# Function to search knowledge from Firebase
def search_knowledge(query, top_k=3):
    knowledge_ref = db.collection("knowledge_base").stream()
    knowledge_data = []

    for doc in knowledge_ref:
        knowledge_data.append(doc.to_dict())

    if not knowledge_data:
        return "No knowledge available."

    contents = [item['content'] for item in knowledge_data]
    corpus_embeddings = embedder.encode(contents, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append(contents[idx])

    return results if results else "No relevant knowledge found."