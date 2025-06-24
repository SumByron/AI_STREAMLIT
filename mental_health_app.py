# mental_health_app.py

import streamlit as st
import os
import numpy as np
import faiss
import openai
from dotenv import load_dotenv

# --- Load API Key ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Config ---
CHUNK_SIZE = 1024
TOP_K = 3
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
DOCUMENT_PATH = "mental_health_data.txt"

# --- Functions ---
@st.cache_data
def load_chunks_from_file(filepath, chunk_size=CHUNK_SIZE):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

@st.cache_resource
def get_text_embedding(text):
    response = openai.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return np.array(response.data[0].embedding)

@st.cache_resource
def create_faiss_index(chunks):
    embeddings = [get_text_embedding(chunk) for chunk in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def get_response_from_context(question, chunks, index):
    question_embedding = get_text_embedding(question)
    D, I = index.search(np.array([question_embedding]), k=TOP_K)
    retrieved_context = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""
    You are a compassionate and informative AI assistant trained to answer mental health questions.
    Use only the following context. Do not rely on outside knowledge.

    Context:
    ---------------------
    {retrieved_context}
    ---------------------

    Question: {question}
    Answer:
    """

    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "You provide accurate, compassionate, mental health support."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# --- Streamlit UI ---
st.title("🧠 Mental Health Assistant")
st.markdown("Ask me anything about mental health. I'm here to help with compassion and evidence-based support.")

if "faiss_index" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        chunks = load_chunks_from_file(DOCUMENT_PATH)
        index, _ = create_faiss_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = index

question = st.text_input("💬 Your question:", placeholder="How do I cope with anxiety?")
if question:
    with st.spinner("Thinking..."):
        answer = get_response_from_context(question, st.session_state.chunks, st.session_state.faiss_index)
    st.success(answer)
