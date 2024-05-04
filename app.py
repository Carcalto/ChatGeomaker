import streamlit as st
from groq import Groq
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

st.set_page_config(page_icon="💬 Prof. Marcelo Claro", layout="wide", page_title="Geomaker Chat Interface")
icon("🌎")

st.subheader("Geomaker Chat Streamlit App")
st.write("Professor Marcelo Claro")

api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else "your_api_key_here"
groq_client = Groq(api_key=api_key)
llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)

# Inicialização do session state para mensagens, se ainda não existir
if 'messages' not in st.session_state:
    st.session_state.messages = []

model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"], index=0)
max_tokens_range = models[model_option]["tokens"]
max_tokens = st.slider("Max Tokens:", min_value=512, max_value=max_tokens_range, value=min(32768, max_tokens_range), step=512)

# Configuração do Faiss Index para documentos
model = SentenceTransformer('all-MiniLM-L6-v2')
d = 384  # Dimension of embeddings
faiss_index = faiss.IndexFlatL2(d)

with st.sidebar:
    st.image("Untitled.png", width=100)
    st.write("Configurações")
    uploaded_files = st.file_uploader("Upload up to 5 documents", accept_multiple_files=True, type=['txt'])
    system_prompt = st.text_area("Define the system prompt", value="Enter a default system prompt or update it dynamically here.")
    if st.button("Confirmar Prompt"):
        st.session_state.system_prompt = system_prompt
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.experimental_rerun()

# Processa e indexa os documentos carregados
if uploaded_files is not None:
    for file in uploaded_files:
        text = file.getvalue().decode("utf-8")
        embedding = model.encode([text])[0]
        faiss_index.add(np.array([embedding]))

def process_chat_with_rag(prompt):
    # Faiss search to enrich response
    query_embedding = model.encode([prompt])
    _, idx = faiss_index.search(query_embedding, k=1)
    related_text = "Related information from documents: " + str(idx)

    messages = [
        ChatMessage(role="system", content=st.session_state.system_prompt),
        ChatMessage(role="user", content=prompt + "\n" + related_text)
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
