import streamlit as st
from groq import Groq  # Assumindo que 'groq' é o módulo correto.
import os

# Configuração inicial da página
st.set_page_config(page_icon="💬", layout="wide", page_title="Chat IA Avançado")
st.image('logo.png', width=100)  # Adicione o logo no cabeçalho

# Conexão com a API Groq
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)

# Função para criar e gerenciar a sessão de chat
def manage_chat_session():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    model_id = st.sidebar.selectbox(
        "Escolha o Modelo:",
        options=["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        index=0,
        format_func=lambda id: id.split('-')[0].capitalize() + " " + id.split('-')[1] + " (" + id.split('-')[2] + ")"
    )

    max_tokens = st.sidebar.slider("Máximo de Tokens:", 512, 32768, 2048)

    prompt = st.text_input("Digite sua pergunta:", key="user_input")
    if st.button("Enviar"):
        system_prompt = f"Modelo: {model_id}, Tokens: {max_tokens}\n"
        # Simulando uma chamada para a API Groq para processar o prompt
        response = f"Resposta simulada para '{prompt}' usando o modelo {model_id}."
        st.session_state.messages.append((prompt, response))

    for idx, (user_question, bot_response) in enumerate(st.session_state.messages):
        with st.container():
            st.text_area(f"Pergunta {idx+1}", value=user_question, height=75, disabled=True)
            st.text_area(f"Resposta {idx+1}", value=bot_response, height=100, disabled=True)

# Chamada principal da função de gerenciamento do chat
manage_chat_session()
