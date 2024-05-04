import streamlit as st
from groq import Groq
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

st.set_page_config(page_icon="💬 Prof. Marcelo Claro", layout="wide", page_title="Geomaker Chat Interface")
icon("🌎")  # Exibe o ícone do globo

st.subheader("Geomaker Chat Streamlit App")
st.write("Professor Marcelo Claro")

api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "your_api_key_here"
groq_client = Groq(api_key=api_key)
llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)

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
    st.image("https://example.com/image.png", width=100)
    st.write("Configurações")
    uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True, type=['txt'], help="Upload up to 5 documents.")

    # Faiss Index initialization
    dimension = 128  # Assuming the vector size from the model
    index = faiss.IndexFlatL2(dimension)
    vectorizer = TfidfVectorizer()
    svd = TruncatedSVD(n_components=dimension)

    # Process and index documents
    if uploaded_files and len(uploaded_files) <= 5:
        documents = [file.getvalue().decode("utf-8") for file in uploaded_files]
        tfidf_matrix = vectorizer.fit_transform(documents)
        reduced_matrix = svd.fit_transform(tfidf_matrix.toarray())
        index.add(reduced_matrix.astype('float32'))

    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.experimental_rerun()

def process_chat_with_rag(prompt):
    # Embed the prompt using similar method as documents
    prompt_vector = svd.transform(vectorizer.transform([prompt]).toarray()).astype('float32')
    _, indices = index.search(prompt_vector, k=1)  # Find the closest document vector
    related_document = documents[indices[0][0]]
    
    messages = [
        ChatMessage(role="system", content=f"Related information: {related_document}"),
        ChatMessage(role="user", content=prompt)
    ]
    response = llama_groq.chat(messages)
    return response

if prompt := st.text_area("Insira sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = process_chat_with_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
