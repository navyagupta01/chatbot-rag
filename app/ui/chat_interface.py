"""
💬 Simplified Chat Interface
============================

Clean chat interface without document upload sections.
"""

import streamlit as st

def render_chat_interface(rag_engine):
    """
    Render clean chat interface without file upload

    Args:
        rag_engine: RAG engine instance
    """

    # Simple example queries if no documents loaded
    if not rag_engine.documents_loaded:
        st.markdown("### 💡 Try these once you upload documents:")

        examples = [
            "📋 Summarize the main findings",
            "🔍 What methodology was used?",
            "📊 Compare the different approaches",
            "🧠 Analyze the key conclusions"
        ]

        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.example_query = example.split(" ", 1)[1]
                    st.rerun()

    # Show system status in main area if documents are loaded
    if rag_engine.documents_loaded:
        status = rag_engine.get_system_status()
        st.success(f"🎯 **Ready!** {status['total_chunks']} chunks from {status.get('embedding_stats', {}).get('total_files', 0)} files processed")
