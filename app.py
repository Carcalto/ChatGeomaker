import streamlit as st
from groq import Groq  # Supondo que 'groq' seja o módulo correto para a API.
import os

# Configuração da página e exibição de ícones personalizados
def icon(emoji: str):
    """Exibe um emoji como ícone no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

st.set_page_config(page_icon="💬", layout="wide", page_title="Chat Interface with Groq")
icon("🧠")

st.subheader("Interactive AI-assisted Chat Application for Education")
st.write("Professor Marcelo Claro")

# Configuração da chave API e tratamento de erro
try:
    api_key = os.environ.get("GROQ_API_KEY", "your_api_key_here")  # Uso de variável de ambiente para a chave API
    groq_client = Groq(api_key=api_key)
except Exception as e:
    st.error(f"Error setting up API: {str(e)}")
    st.stop()

# Modelos disponíveis
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-Instruct", "tokens": 8192, "developer": "Facebook"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-chat", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"}
}

# Seleção do modelo na interface
model_option = st.selectbox("Choose a model:", options=list(models.keys()), format_func=lambda x: models[x]["name"])

# Configuração adicional para tokens e prompt
max_tokens = st.slider("Max Tokens:", min_value=512, max_value=models[model_option]["tokens"], value=2048, step=512)
system_prompt = st.text_area("Set System Prompt:")

# Gerenciamento de estado da sessão para armazenar mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Função para processar chat com RAG para melhorar a qualidade das respostas
def process_chat_with_rag(prompt):
    try:
        # Suposição de como enviar uma requisição usando Groq com RAG
        response = groq_client.query(prompt, model_id=models[model_option]["name"], max_tokens=max_tokens)
        return response
    except Exception as e:
        return f"Error processing response: {str(e)}"

# Interface para entrada de perguntas
user_question = st.text_input("Enter your question here:")
if user_question:
    response = process_chat_with_rag(user_question + " " + system_prompt)  # Concatena a pergunta do usuário com o prompt do sistema
    st.session_state.messages.append({"user": user_question, "bot": response})
    st.write("Chatbot:", response)

# Exibição de mensagens anteriores
for message in st.session_state.messages:
    st.write("You:", message["user"])
    st.write("Chatbot:", message["bot"])
