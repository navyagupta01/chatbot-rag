import streamlit as st
import os
from dotenv import load_dotenv

from core.rag_engine import AgenticRAGEngine
from ui.sidebar import render_sidebar
from ui.chat_interface import render_chat_interface
from config import AppConfig

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

def handle_basic_greetings(message: str) -> str:
    """Handle basic greetings and common questions without PDFs"""
    message_lower = message.lower().strip()

    greetings = {
        'hi': "Hello! 👋 I'm your Agentic RAG assistant. Upload some PDF documents in the sidebar and I can help you analyze them!",
        'hello': "Hi there! 😊 I'm here to help you understand your documents. Just upload some PDFs in the sidebar to get started!",
        'hey': "Hey! 🤖 Ready to dive into some document analysis? Upload your PDFs and ask me anything about them!",
        'how are you': "I'm doing great! I'm an AI assistant specialized in analyzing PDF documents. How can I help you today?",
        'what can you do': "I can help you analyze PDF documents! I can summarize content, answer specific questions, compare information, and provide detailed analysis. Just upload your PDFs in the sidebar!",
        'help': "Here's what I can do:\n📄 Analyze PDF documents\n📋 Summarize content\n🔍 Answer specific questions\n📊 Compare information\n🧠 Provide detailed analysis\n\nUpload your PDFs in the sidebar to get started!"
    }

    for key, response in greetings.items():
        if key in message_lower:
            return response

    return None

def main():
    """Main application function"""

    # Page configuration
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Clean main header
    st.title("🤖 Agentic RAG PDF Chatbot")
    st.markdown("*Upload PDFs in the sidebar and ask me anything about them!*")

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        st.error("🔑 **API Key Required**: Please set your `OPENROUTER_API_KEY` in your `.env` file")
        st.info("💡 **Tip**: Copy `.env.example` to `.env` and add your OpenRouter API key")
        return

    # Initialize RAG engine
    if st.session_state.rag_engine is None:
        try:
            st.session_state.rag_engine = AgenticRAGEngine(
                api_key=api_key,
                model=AppConfig.DEFAULT_MODEL
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG engine: {str(e)}")
            return

    # Render sidebar (file upload only)
    render_sidebar(st.session_state.rag_engine)

    # Main chat interface
    st.markdown("---")

    # Chat input
    user_input = st.chat_input("Ask me anything! Try saying 'hi' or upload PDFs for document analysis...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Check for basic greetings first
        greeting_response = handle_basic_greetings(user_input)

        if greeting_response:
            # Handle basic greetings
            st.session_state.chat_history.append({"role": "assistant", "content": greeting_response})

        elif st.session_state.rag_engine.documents_loaded:
            # Process with RAG if documents are loaded
            with st.spinner("🤔 Analyzing your question..."):
                result = st.session_state.rag_engine.query(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": result.answer})

        else:
            # No documents loaded
            response = "I'd love to help you with that! However, I need some PDF documents to analyze first. Please upload your PDFs using the sidebar, and then I can provide detailed answers based on your documents. 📄"
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Welcome message if no chat history
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown("👋 **Welcome!** I'm your intelligent document assistant. I can help you:")
            st.markdown("- 📋 Summarize document content")
            st.markdown("- 🔍 Answer specific questions")
            st.markdown("- 📊 Compare information")
            st.markdown("- 🧠 Provide detailed analysis")
            st.markdown("\n**Get started by uploading PDFs in the sidebar!** ➡️")

if __name__ == "__main__":
    main()
