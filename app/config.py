
import os
from typing import Dict, List

class AppConfig:
    """Main application configuration with optimized settings for Mistral"""

    # 🎨 UI Configuration
    PAGE_TITLE = "Agentic RAG PDF Chatbot"
    PAGE_ICON = "🤖"

    # 📄 PDF Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_FILE_SIZE_MB = 200

    # 🧠 Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = 384

    # 🔗 API Configuration
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-7b-instruct")

    # 🎯 RAG Configuration - LOWERED THRESHOLDS
    DEFAULT_TOP_K = 5
    MIN_SIMILARITY_THRESHOLD = 0.1  # ⬅️ Lowered from 0.3

    # 📁 Data Paths
    DATA_DIR = "data"
    UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
    INDEXES_DIR = os.path.join(DATA_DIR, "indexes")

    # 🤖 Available Models - UPDATED FOR MISTRAL
    AVAILABLE_MODELS = {
        "Mistral 7B Instruct": "mistralai/mistral-7b-instruct",
        "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct",
        "GPT-3.5 Turbo": "openai/gpt-3.5-turbo",
        "GPT-4": "openai/gpt-4",
        "Claude Instant": "anthropic/claude-instant-v1"
    }

    # 📝 Example Queries
    EXAMPLE_QUERIES = [
        "📋 Summarize the main points of this document",
        "🔍 What methodology was used in this research?",
        "📊 Compare different approaches mentioned in the document",
        "🧠 Analyze the key findings and their implications",
        "❓ What are the limitations discussed?",
        "🎯 What recommendations are provided?"
    ]

class QueryTypes:
    """Query type definitions for agentic planning"""
    SUMMARY = "summary"
    SEARCH = "search"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"

class AgentConfig:
    """Configuration for agentic behavior - ALL THRESHOLDS LOWERED"""

    # Planning thresholds for different query types
    QUERY_TYPE_THRESHOLDS = {
        QueryTypes.SUMMARY: {
            "chunks_needed": 10,
            "similarity_threshold": 0.1,
            "max_tokens": 1500
        },
        QueryTypes.SEARCH: {
            "chunks_needed": 5,
            "similarity_threshold": 0.1,
            "max_tokens": 800
        },
        QueryTypes.COMPARISON: {
            "chunks_needed": 8,
            "similarity_threshold": 0.1,
            "max_tokens": 1200
        },
        QueryTypes.ANALYSIS: {
            "chunks_needed": 7,
            "similarity_threshold": 0.1,
            "max_tokens": 1300
        }
    }

    # Keywords for query type detection
    QUERY_KEYWORDS = {
        QueryTypes.SUMMARY: ["summarize", "summary", "overview", "main points", "key points", "sum-up","brief"],
        QueryTypes.SEARCH: ["find", "search", "what", "where", "when", "who", "how"],
        QueryTypes.COMPARISON: ["compare", "comparison", "versus", "vs", "difference", "similarities"],
        QueryTypes.ANALYSIS: ["analyze","analyse", "analysis", "explain", "why", "implications", "significance"]
    }
