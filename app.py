import streamlit as st
import os
from groq import Groq
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessage, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import ChatMessage
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def load_image(image_path):
    """Lê a imagem do disco e retorna como bytes."""
    with open(image_path, "rb") as file:
        return file.read()

def main():
    # Configuração da página
    st.set_page_config(page_icon="💬", layout="wide", page_title="Chat Inteligente com Suporte a Memória e RAG")
    st.markdown(f'<span style="font-size: 78px;">🧠</span>', unsafe_allow_html=True)  # Ícone grande
    st.title("Aplicativo de Chat Avançado para Educação")
    st.write("Bem-vindo ao sistema avançado de chat!")

    # Carregar e exibir o logo
    image_path = 'path/to/Untitled.png'
    image_data = load_image(image_path)
    st.image(image_data, width=100)

    # Configurações de ambiente e API
    api_key = st.secrets.get("GROQ_API_KEY", "your_api_key_here")
    groq_client = Groq(api_key=api_key)
    llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)

    # Preparação da memória de conversação
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    memory = MessagesPlaceholder(variable_name="chat_history")

    # Configurações de modelo e prompt
    system_prompt = st.text_area("Defina o prompt do sistema:", "Digite aqui...")
    model_choice = st.selectbox("Escolha um modelo:", ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])

    # Entrada de pergunta do usuário
    user_question = st.text_input("Insira sua pergunta aqui:")
    if user_question:
        # Adicionando a pergunta à memória
        memory.save_context({'input': user_question}, {'output': ''})

        # Criação do prompt
        prompt_template = ChatPromptTemplate([
            SystemMessage(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate(template="{human_input}")
        ])
        conversation_chain = LLMChain(
            llm=GroqLLM(api_key=api_key, model_name=model_choice),
            prompt=prompt_template,
            memory=memory
        )

        # Previsão e resposta
        response = conversation_chain.predict(human_input=user_question)
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.write("Resposta do Chatbot:", response)

    # Exibição do histórico de mensagens
    for message in st.session_state.chat_history:
        role = "🤖" if message['role'] == 'assistant' else "👤"
        st.write(f"{role} {message['content']}")

if __name__ == "__main__":
    main()
