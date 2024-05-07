import streamlit as st
import os
import pandas as pd
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def upload_json_data():
    """
    Permite aos usuários fazer upload de arquivos JSON que podem ser usados como fonte de dados.
    Os arquivos são carregados através de um widget de upload no Streamlit.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos JSON (até 2 arquivos, 300MB cada)", type='json', accept_multiple_files=True, key="json_upload")
    if uploaded_files:
        data_frames = [pd.read_json(file) for file in uploaded_files]
        st.session_state['uploaded_data'] = data_frames
        for i, df in enumerate(data_frames, 1):
            st.write(f"Pré-visualização do arquivo JSON {i} carregado:")
            st.dataframe(df.head())
    return uploaded_files

def main():
    """
    Função principal que configura a interface do usuário e o fluxo de interação do chatbot.
    """
    # Configura o ícone da página, layout e título para o aplicativo Streamlit.
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG")
    
    # Mostra o logotipo para reforçar a marca ou a identidade visual do serviço de chat.
    st.image("caminho_para_seu_logo.png", width=100)  # Substitua pelo caminho correto do arquivo de imagem
    st.title("Bem-vindo ao Chat Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    # Obtém a chave API a partir das variáveis de ambiente.
    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')

    # Configura opções de personalização na barra lateral do Streamlit.
    st.sidebar.title('Customização')
    primary_prompt = st.sidebar.text_input("Prompt do sistema principal", "Como posso ajudar você hoje?")
    secondary_prompt = st.sidebar.text_input("Prompt do sistema secundário", "Há algo mais em que posso ajudar?")
    model_choice = st.sidebar.selectbox("Escolha um modelo", ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
    conversational_memory_length = st.sidebar.slider('Tamanho da memória conversacional', 1, 10, value=5)

    # Configura a memória conversacional para armazenar o histórico de interações.
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Inicializa o cliente de chat usando a API Groq.
    groq_chat = ChatGroq(api_key=groq_api_key, model_name=model_choice)

    # Permite o upload de arquivos JSON para serem usados como dados de referência.
    uploaded_files = upload_json_data()
    
    # Captura a entrada do usuário para a pergunta.
    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        # Alternância dinâmica dos prompts para manter a conversa interessante.
        current_prompt = secondary_prompt if 'last_prompt' in st.session_state and st.session_state.last_prompt == primary_prompt else primary_prompt
        st.session_state.last_prompt = current_prompt

        # Constrói o prompt usando a mensagem do sistema e o histórico de chat.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=current_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ])

        # Cria uma cadeia LLM para processar a conversa usando o modelo selecionado.
        conversation = LLMChain(llm=groq_chat, prompt=prompt, memory=memory)

        # Gera a resposta utilizando o modelo de linguagem.
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        
        # Atualiza o histórico de conversas armazenado na memória.
        st.session_state.chat_history.append(message)
        
        # Exibe a resposta do chatbot na interface.
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
