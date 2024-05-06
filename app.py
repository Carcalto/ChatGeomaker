import streamlit as st
import os
from groq import Groq  # Certifique-se de que o Groq está corretamente importado.
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage
from langchain.chains import LLMChain
from langchain.llms import GroqLLM
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessage, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

def main():
    # Configurações iniciais da página
    st.set_page_config(page_icon="💬", layout="wide", page_title="Chat Avançado com Memória e RAG")
    st.markdown(f'<span style="font-size: 78px;">🧠</span>', unsafe_allow_html=True)  # Ícone grande
    st.title("Aplicativo de Chat Avançado para Educação")
    st.write("Bem-vindo ao sistema avançado de chat!")

    # Configurações de ambiente e API
    api_key = st.secrets.get("GROQ_API_KEY", "your_api_key_here")
    groq_client = Groq(api_key=api_key)
    llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)

    # Preparação da memória de conversação
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    memory = ConversationBufferWindowMemory(k=5)

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
