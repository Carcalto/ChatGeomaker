import streamlit as st
from groqcloud import GroqCloud  # Supõe-se que esse módulo exista e esteja corretamente configurado

# Configuração inicial da página
st.set_page_config(page_icon="💬", layout="wide", page_title="Aplicativo de Chat Interativo Avançado")

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

icon("🧠")

# Subtítulo e cabeçalho da aplicação
st.subheader("Aplicativo de Chat Assistido por IA")

# Instanciação do cliente GroqCloud com a API Key
api_key = st.secrets.get("GROQ_API_KEY", "your_api_key_here")
client = GroqCloud(api_key=api_key)

# Definição dos modelos de linguagem disponíveis
models = {
    "llama3-8b-8192": {
        "name": "LLaMA3-8b-chat",
        "tokens": 8192,
        "developer": "Meta",
    },
    "llama3-70b-8192": {
        "name": "LLaMA3-70b-chat",
        "tokens": 8192,
        "developer": "Meta",
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral-8x7b-Instruct-v0.1",
        "tokens": 32768,
        "developer": "Mistral",
    },
    "gemma-7b-it": {
        "name": "Gemma-7b-it",
        "tokens": 8192,
        "developer": "Google",
    }
}

# Seleção de modelo pelo usuário
model_option = st.selectbox(
    "Escolha um modelo:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"]
)

max_tokens_range = models[model_option]["tokens"]

# Slider para definir o máximo de tokens
max_tokens = st.slider(
    "Máximo de Tokens:",
    min_value=512,
    max_value=max_tokens_range,
    value=min(max_tokens_range, max_tokens_range),
    step=512
)

# Processamento e exibição do chat
if prompt := st.text_area("Insira sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Chamada à API para obter respostas do modelo selecionado
    chat_completion = client.chat.completions.create(
        model=model_option,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        max_tokens=max_tokens,
        stream=True,
    )

    # Processar e exibir respostas
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            response = chunk.choices[0].delta.content
            st.session_state.messages.append({"role": "assistant", "content": response})

# Exibição das mensagens de chat
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
