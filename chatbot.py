import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer, util

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load GPT-J-6B model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/gpt-j-6B", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).cuda()

# Load Sentence Transformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate Chatbot Response using GPT-J-6B
def generate_response(prompt):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        output = model.generate(input_ids, max_length=200, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Save Knowledge to Firestore
def save_knowledge(title, content):
    # Save the document under the "knowledge_base" collection
    doc_ref = db.collection("knowledge_base").document(title)
    doc_ref.set({"content": content})
    print(f"Knowledge saved: {title}")

# Retrieve Knowledge for Semantic Search
def search_knowledge(query):
    # Fetch all documents from Firestore under the "knowledge_base" collection
    docs = db.collection("knowledge_base").stream()
    texts = []
    titles = []

    for doc in docs:
        data = doc.to_dict()
        texts.append(data['content'])
        titles.append(doc.id)

    if not texts:
        return None

    # Embedding and Semantic Search
    query_vec = embedder.encode(query, convert_to_tensor=True)
    text_vecs = embedder.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(query_vec, text_vecs)[0]
    best_idx = torch.argmax(scores).item()
    
    # Return the most relevant document
    return texts[best_idx]