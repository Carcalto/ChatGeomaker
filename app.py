import streamlit as st
import os
from groq import Groq  # Substitua 'groq' pelo módulo correto se necessário.
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage

from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    """Ponto de entrada principal do aplicativo de chat com IA."""
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Geomaker")
    
    # Configuração e exibição de logotipo
    st.image("path_to_groq_logo.png", width=100)  # Substitua pelo caminho correto da imagem
    st.title("Bem-vindo ao Chat com Groq!")
    st.write("Olá! Sou o seu chatbot Groq amigável. Posso ajudar a responder suas perguntas, fornecer informações ou apenas conversar.")

    # Obter chave API do ambiente
    groq_api_key = os.getenv('GROQ_API_KEY', 'Sua_chave_API_padrão')

    # Opções de personalização na barra lateral
    system_prompt = st.sidebar.text_input("Prompt do sistema:")
    model_choice = st.sidebar.selectbox("Escolha um modelo:", ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional:', 1, 10, value=5)

    # Gerenciamento de memória conversacional
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Inicializar cliente de chat Groq
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)

        # Gerar resposta do chatbot
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
