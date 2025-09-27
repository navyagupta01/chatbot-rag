#!/bin/bash
# Quick setup script for Agentic RAG PDF Chatbot

echo "🤖 Setting up Agentic RAG PDF Chatbot..."

# Create directory structure
mkdir -p data/{uploads,embeddings,indexes}
mkdir -p examples/sample_pdfs

# Create __init__.py files
touch app/__init__.py
touch app/core/__init__.py
touch app/agents/__init__.py
touch app/ui/__init__.py
touch app/utils/__init__.py

# Create .gitkeep files for empty directories
touch data/uploads/.gitkeep
touch data/embeddings/.gitkeep
touch data/indexes/.gitkeep
touch examples/sample_pdfs/.gitkeep

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete! Run with: streamlit run app/main.py"
