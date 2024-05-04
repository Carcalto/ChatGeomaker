import streamlit as st
from groq import Groq  # Assumindo que esta é a API correta, ajuste conforme necessário

# Configuração inicial da página Streamlit
st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Avançada com RAG")
icon = lambda emoji: st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

# Exibe um ícone e título personalizados
icon("🧠")
st.subheader("Aplicativo de Chat Avançado")

# Inicializa a API com a chave de segurança
api_key = "sua_chave_api"  # Substitua com a chave real ou use st.secrets para maior segurança
client = Groq(api_key=api_key)

# Define os detalhes do modelo e a capacidade máxima de tokens
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-Instruct", "tokens": 8192, "developer": "OpenAI"},
    "gpt3-175b-rag": {"name": "GPT-3 175B with RAG", "tokens": 4096, "developer": "OpenAI RAG"}
}
model_option = st.selectbox("Escolha um modelo:", options=list(models.keys()), format_func=lambda x: models[x]["name"])

# Inicialização e gestão do estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# Permite ao usuário definir um prompt de sistema inicial
system_prompt = st.text_area("Defina o prompt inicial do sistema:", value=st.session_state.system_prompt)
if st.button("Confirmar Prompt"):
    st.session_state.system_prompt = system_prompt

# Área para inserção de perguntas com opção de limpar a conversa
if st.button("Limpar Conversa"):
    st.session_state.messages = []
prompt = st.text_input("Insira sua pergunta aqui...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[{"role": "system", "content": st.session_state.system_prompt}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            max_tokens=models[model_option]["tokens"],
            stop_sequences=["\n"],
            return_prompt=True
        )
        # Adiciona a resposta ao estado da sessão
        response = chat_completion.choices[0].text.strip()
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Erro ao obter resposta: {str(e)}")

# Exibição das mensagens anteriores
for message in st.session_state.messages:
    role, content = message['role'], message['content']
    avatar = "🤖" if role == "assistant" else "👨‍💻"
    with st.container():
        st.markdown(f"{avatar} **{role.capitalize()}**: {content}")

# A caixa de entrada para o chat
if st.text_area("Enviar nova mensagem:"):
    # Processamento adicional aqui
    pass
