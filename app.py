import streamlit as st
from groq import Groq  # Verifique se 'groq' é o módulo correto.
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.core.llms import ChatMessage

def icon(emoji: str):
    """Mostra um emoji como ícone de página no estilo Notion."""
    st.write(f'<span style="font-size: 78px; line-height: 1">{emoji}</span>', unsafe_allow_html=True)

# Configuração da página com mais opções de personalização
st.set_page_config(page_icon="💬", layout="wide", page_title="Interface de Chat Geomaker")
icon("")  # Exibe um ícone personalizado

st.subheader("Aplicativo de Chat Geomaker")
st.write("Professor Marcelo Claro")

# Configuração da API Key com tratamento de erro
try:
    api_key = st.secrets.get("GROQ_API_KEY", "your_api_key_here")
    groq_client = Groq(api_key=api_key)
    llama_groq = LlamaGroq(model="llama3-70b-8192", api_key=api_key)
except Exception as e:
    st.error(f"Erro ao configurar a API: {str(e)}")
    st.stop()

# Inicialização do estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'show_manual' not in st.session_state:
    st.session_state.show_manual = False  # Manual inicialmente oculto

# Configuração dos modelos disponíveis com descrição detalhada
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b-Instruct", "tokens": 32768, "developer": "Facebook"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-chat", "tokens": 32768, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 32768, "developer": "Google"}
}

model_option = st.selectbox("Escolha um modelo:", options=list(models.keys()), format_func=lambda x: models[x]["name"])
max_tokens_range = models[model_option]["tokens"]
max_tokens = st.slider("Máximo de Tokens:", min_value=512, max_value=max_tokens_range, value=min(32768, max_tokens_range), step=512)

# Configurações adicionais na barra lateral com melhor navegação
with st.sidebar:
    st.image("Untitled.png", width=100)
    st.write("Configurações")
    if st.button("Mostrar/Ocultar Manual de Uso"):
        st.session_state.show_manual = not st.session_state.show_manual

    if st.session_state.show_manual:
        # Manual de Uso detalhado aqui
        st.write("## Manual de Uso")
                       
        # Introdução
        st.write("""
        ### Introdução
        Bem-vindo ao Aplicativo de Chat Geomaker! Este aplicativo permite interagir com modelos avançados de linguagem artificial para gerar respostas baseadas em suas perguntas.
        """)

        # Como Iniciar
        st.write("""
        ### Como Iniciar
        - Acesse a interface principal.
        - Visualize o menu de seleção de modelos na parte superior da tela.
        """)

        # Escolha de Modelos
        st.write("""
        ### Escolha de Modelos
        - Use o dropdown para selecionar o modelo de linguagem desejado.
        - Cada modelo possui uma descrição de suas capacidades e limitações.
        """)

        # Envio de Mensagens
        st.write("""
        ### Envio de Mensagens
        - Digite sua pergunta na área 'Insira sua pergunta aqui...'.
        - Clique em enviar para ver a resposta do modelo selecionado.
        """)

        # Ajustes de Configuração
        st.write("""
        ### Ajustes de Configuração
        - Ajuste o número máximo de tokens que o modelo deve processar usando o slider abaixo da seleção do modelo.
        """)

        # Uso Avançado
        st.write("""
        ### Uso Avançado
        - Explore funcionalidades avançadas acessando as configurações no canto inferior da barra lateral.
        """)

        # Resolução de Problemas
        st.write("""
        ### Resolução de Problemas
        Se encontrar problemas, reinicie o aplicativo ou entre em contato com o suporte técnico.
        """)

        # Feedback e Melhorias
        st.write("""
        ### Feedback e Melhorias
        - Sua opinião é importante para nós! Use o formulário de feedback disponível na aba de configurações para enviar suas sugestões.
        """)


    # Campo para definir o prompt do sistema
    system_prompt = st.text_area("Defina o prompt do sistema:")
    if st.button("Confirmar Prompt"):
        st.session_state.system_prompt = system_prompt
    if st.button("Limpar Conversa"):
        st.session_state.messages = []
        st.experimental_rerun()

# Processamento de chat com RAG com validação
def process_chat_with_rag(prompt):
    try:
        messages = [
            ChatMessage(role="system", content=st.session_state.system_prompt),
            ChatMessage(role="user", content=prompt)
        ]
        response = llama_groq.chat(messages)
        return response
    except Exception as e:
        return f"Erro ao processar a resposta: {str(e)}"

# Área para inserção de perguntas
if prompt := st.text_area("Insira sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = process_chat_with_rag(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Exibição das mensagens
for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
