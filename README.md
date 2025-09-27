# 🤖 Agentic RAG PDF Chatbot

An intelligent document analysis assistant that combines **Retrieval-Augmented Generation (RAG)** with **Agentic AI** to process PDF documents and answer complex queries with step-by-step reasoning.

## 🎯 Key Features

- **📄 Dynamic PDF Processing**: Upload any PDF - no pre-training required
- **🧠 Agentic Reasoning**: Plans approach based on query type
- **🔍 Semantic Search**: Uses embeddings for intelligent information retrieval
- **💬 Contextual Responses**: Generates answers using retrieved document context
- **🎨 Interactive UI**: Clean Streamlit interface with reasoning transparency

## 🚀 Quick Start

main.py-- load .env
initialise session
pdf-- processor--embeddings
chat-- query

# Agentic RAG Chatbot

This project parses a kmsdb_dump.dump file, extracts all links and data, and answers questions using a simple retrieval-augmented generation (RAG) pipeline.

## Files
- parser.py: Extracts links and data from the dump file.
- rag_pipeline.py: Answers questions using the parsed data.
- requirements.txt: Project dependencies.

## Usage
1. Place kmsdb_dump.dump in the project directory.
2. Run parser.py to extract links.
3. Run rag_pipeline.py to ask questions about the data.

## Next Steps
- Integrate embeddings and advanced retrieval for better answers.
- Add support for more file formats and data types.

