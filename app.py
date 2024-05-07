import streamlit as st
from groq import Groq  # Verifique a disponibilidade e correção deste import.
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Geomaker")
icon("🧠")

st.subheader("Aplicativo de Chat assistida por IA para Educação")
st.write("Professor Marcelo Claro")

try:
    api_key = st.secrets["GROQ_API_KEY"]  # Correção para acesso correto ao segredo.
    groq_client = Groq(api_key=api_key)
    llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)
except Exception as e:
    st.error(f"Erro ao configurar a API: {str(e)}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'show_manual' not in st.session_state:
    st.session_state.show_manual = False

models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-Instruct", "tokens": 32768, "developer": "Facebook"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-chat", "tokens": 32768, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 32768, "developer": "Google"}
}

model_option = st.selectbox("Escolha um modelo:", options=list(models.keys()), format_func=lambda x: models[x]["name"])
max_tokens_range = models[model_option]["tokens"]
max_tokens = st.slider("Máximo de Tokens:", min_value=512, max_value=max_tokens_range, value=min(32768, max_tokens_range), step=512)

with st.sidebar:
    st.image("Untitled.png", width=100)
    st.write("Configurações")
    if st.button("Mostrar/Ocultar Manual de Uso"):
        st.session_state.show_manual = not st.session_state.show_manual

    if st.session_state.show_manual:
        st.write("## Manual de Uso")
        # Manual de Uso detalhado aqui

    system_prompt = st.text_area("Defina o prompt do sistema: - Busque o catalogo de prompt para educador.")
    if st.button("Confirmar Prompt"):
        st.session_state.system_prompt = system_prompt
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.experimental_rerun()
    st.image("eu.ico", width=100)
    st.write("Projeto Geomaker + IA - Professor: Marcelo Claro.")

def process_chat_with_rag(prompt):
    try:
        messages = [ChatMessage(role="system", content=st.session_state.system_prompt), ChatMessage(role="user", content=prompt)]
        response = llama_groq.chat(messages)
        return response
    except Exception as e:
        return f"Erro ao processar a resposta: {str(e)}"

if prompt := st.text_area("Insira sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = process_chat_with_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
