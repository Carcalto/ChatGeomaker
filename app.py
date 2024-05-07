import streamlit as st
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

def main():
    """Main function to set up and handle the AI-powered chat interface."""
    # Set up page configuration and display initial information
    st.set_page_config(page_icon="🤖", layout="wide", page_title="AI Chat Interface")
    st.title("Welcome to the AI-powered Chat!")
    st.write("Please ask your questions below:")

    # Environment setup for API key
    groq_api_key = os.getenv('GROQ_API_KEY', 'default_api_key_if_not_set')
    if groq_api_key == 'default_api_key_if_not_set':
        st.error("API Key is not set in the environment variables.")
        st.stop()

    # Setup ChatGroq client
    model_choice = st.sidebar.selectbox("Select your model:", ["llama3-8b-8192", "gemma-7b-it"])
    chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name=model_choice)

    # Define conversation memory
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    # Handling user input
    user_input = st.text_input("Your question:")

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Manage chat history and context
    for message in st.session_state.chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Construct prompt template
    prompt_template = ChatPromptTemplate(
        parts=[
            SystemMessage("Hello, I'm here to help you. Please ask your question."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}")
        ]
    )

    # Create a conversation chain with the LLM and the prompt template
    conversation = LLMChain(llm=chat_groq, prompt=prompt_template, memory=memory)

    # Process the user's question
    if user_input:
        response = conversation.predict(human_input=user_input)
        message = {'human': user_input, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("AI:", response)

if __name__ == "__main__":
    main()
