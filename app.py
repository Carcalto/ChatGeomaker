import streamlit as st
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from groq import Groq
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

st.set_page_config(page_icon="💬 Prof. Marcelo Claro", layout="wide", page_title="Geomaker Chat Interface")
icon("")  # Exibe o ícone do globo

st.subheader("Geomaker Chat Streamlit App")
st.write("Professor Marcelo Claro")

api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "your_api_key_here"
groq_client = Groq(api_key=api_key)
llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-Instruct", "tokens": 32768, "developer": "Facebook"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-chat", "tokens": 32768, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 32768, "developer": "Google"}
}

model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
max_tokens_range = models[model_option]["tokens"]
max_tokens = st.slider("Max Tokens:", min_value=512, max_value=max_tokens_range, value=min(32768, max_tokens_range), step=512, help=f"Adjust the maximum number of tokens for the model's response: {max_tokens_range}")

with st.sidebar:
    st.image("Untitled.png", width=100)
    st.write("Configurações")
    uploaded_files = st.file_uploader("Upload up to 5 documents", accept_multiple_files=True, type=['txt'])
    system_prompt = st.text_area("Define the system prompt", value="Enter a default system prompt or update it dynamically here.")

    # Prepare Faiss index for uploaded documents
    if uploaded_files:
        vectorizer = TfidfVectorizer()
        svd = TruncatedSVD(n_components=128)  # Dimensionality reduction to reduce vector size
        documents = [file.getvalue().decode("utf-8") for file in uploaded_files]
        tfidf_matrix = vectorizer.fit_transform(documents)
        embeddings = svd.fit_transform(tfidf_matrix.toarray())
        index = faiss.IndexFlatL2(128)  # Using L2 distance for similarity
        index.add(np.array(embeddings).astype('float32'))

    if st.button("Confirm Prompt"):
        st.session_state.system_prompt = system_prompt
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

def process_chat_with_rag(prompt):
    prompt_vec = svd.transform(vectorizer.transform([prompt]).toarray()).astype('float32')
    D, I = index.search(prompt_vec, 1)  # Search for the most similar document
    related_doc = documents[I[0][0]] if I.size > 0 else "No related document found."

    messages = [
        ChatMessage(role="system", content=f"Related document: {related_doc}"),
        ChatMessage(role="user", content=prompt)
    ]
    response = llama_groq.chat(messages)
    return response

if prompt := st.text_area("Enter your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = process_chat_with_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
