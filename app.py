import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.llms import groq
from langchain.prompts import SimplePrompt
from langchain.memory import ConversationMemory

def upload_data():
    """
    Permite aos usuários fazer upload de arquivos JSON e CSV, que podem ser usados como fonte de dados.
    Os arquivos são carregados e lidos em DataFrames que são armazenados na sessão.
    """
    uploaded_files = st.file_uploader("Faça upload dos seus arquivos (JSON ou CSV, até 300MB cada)",
                                      type=['json', 'csv'], accept_multiple_files=True)
    data_frames = []
    if uploaded_files:
        for file in uploaded_files:
            try:
                data = pd.read_json(file) if file.type == "application/json" else pd.read_csv(file)
                data_frames.append(data)
                st.session_state[file.name] = data  # Armazena dados na sessão sob o nome do arquivo
                st.write(f"Pré-visualização do arquivo {file.name} carregado:")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Erro ao processar o arquivo {file.name}: {e}")
    return data_frames

def vectorize_text(data):
    """Vetoriza o texto usando TF-IDF para permitir comparações de similaridade."""
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data), vectorizer

def perform_advanced_rag(question, data_frames):
    """
    Executa uma geração aumentada por recuperação mais avançada utilizando vetorização de texto e
    similaridade de cosseno para encontrar dados relevantes.
    """
    if not data_frames:
        return "Nenhum dado disponível para responder à pergunta."

    combined_text = []
    for df in data_frames:
        combined_text += df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    
    tfidf_matrix, vectorizer = vectorize_text(combined_text)
    question_vec = vectorizer.transform([question])

    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    most_relevant_index = similarities.argmax()
    if similarities[most_relevant_index] > 0.1:  # Threshold for relevance
        return f"Com base nos dados carregados: {combined_text[most_relevant_index]}"
    else:
        return "Nenhum dado relevante encontrado para responder à pergunta."

def main():
    st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançado com RAG e Upload de Dados")
    st.image("caminho_para_seu_logo.png", width=100)
    st.title("Bem-vindo ao Chat Avançado com RAG!")
    st.write("Este chatbot utiliza um modelo avançado que combina geração de linguagem com recuperação de informações.")

    groq_api_key = os.getenv('GROQ_API_KEY', 'Chave_API_Padrão')
    model = Groq(api_key=groq_api_key, model_name="your_model_name_here")
    memory = ConversationMemory()
    data_frames = upload_data()

    user_question = st.text_input("Faça uma pergunta:")
    if user_question:
        rag_response = perform_advanced_rag(user_question, data_frames)
        st.write("Resposta RAG:", rag_response)

if __name__ == "__main__":
    main()
