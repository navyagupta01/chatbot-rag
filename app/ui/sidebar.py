"""
📁 Simplified Sidebar - PDF Upload Only
=======================================

Clean sidebar focusing only on PDF upload and file management.
"""

import streamlit as st
from typing import List

def render_sidebar(rag_engine):
    """
    Render simplified sidebar with only PDF upload functionality

    Args:
        rag_engine: RAG engine instance
    """

    with st.sidebar:
        st.header("📄 Document Upload")

        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload PDF documents to analyze"
        )

        # Process uploaded files
        if uploaded_files:
            # Update session state with new files
            st.session_state.uploaded_files = uploaded_files

            # Process button
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    success = rag_engine.load_documents(uploaded_files)
                    if success:
                        st.success("✅ Documents processed!")
                        st.rerun()

        # Show uploaded files
        st.markdown("### 📋 Uploaded Files")

        if hasattr(st.session_state, 'uploaded_files') and st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.markdown(f"📄 **{file.name}** ({file_size_mb:.1f} MB)")
        else:
            st.markdown("*No files uploaded yet*")

        # Document status
        if rag_engine.documents_loaded:
            status = rag_engine.get_system_status()
            st.markdown("### ✅ Ready for Analysis")
            st.info(f"📊 {status['total_chunks']} chunks ready from {status.get('embedding_stats', {}).get('total_files', 0)} files")

        # Clear documents option
        if rag_engine.documents_loaded:
            st.markdown("---")
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                rag_engine.clear_documents()
                st.session_state.uploaded_files = []
                st.rerun()

        # Help section
        st.markdown("---")
        with st.expander("💡 Tips"):
            st.markdown("""
                **For best results:**
                - Use text-based PDFs (not scanned images)
                - Upload multiple related documents
                - Try different types of questions:
                  - "Summarize the main points"
                  - "What methodology was used?"
                  - "Compare the different approaches"
            """)
