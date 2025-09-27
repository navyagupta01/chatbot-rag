
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .pdf_processor import PDFProcessor, DocumentChunk
from .embeddings import EmbeddingManager
from .llm_interface import LLMInterface, LLMResponse
from agents.query_planner import QueryPlanner, QueryPlan
from agents.reasoning_engine import ReasoningEngine, ReasoningStep
from config import AppConfig, AgentConfig

@dataclass
class RAGResult:
    """
    Comprehensive result from the RAG pipeline
    Contains the answer, reasoning process, and metadata
    """
    answer: str                           # Generated response
    query_plan: QueryPlan                # How the agent planned to answer
    reasoning_steps: List[ReasoningStep]  # Step-by-step reasoning
    retrieved_chunks: List[Tuple[DocumentChunk, float]]  # Source chunks with scores
    response_metadata: Dict[str, Any]     # Additional metadata
    processing_time: float                # Total time taken

class AgenticRAGEngine:
    """
    Main RAG Engine with Agentic Capabilities

    This class orchestrates the entire pipeline:
    1. Document processing and indexing
    2. Query analysis and planning (Agentic)
    3. Information retrieval (RAG)
    4. Response generation with reasoning (Agentic + RAG)
    """

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the Agentic RAG Engine

        Args:
            api_key: OpenRouter API key for LLM access
            model: LLM model to use (defaults to config default)
        """
        self.api_key = api_key
        self.model = model or AppConfig.DEFAULT_MODEL

        # Initialize all components
        self.pdf_processor = PDFProcessor(
            chunk_size=AppConfig.CHUNK_SIZE,
            overlap=AppConfig.CHUNK_OVERLAP
        )

        self.embedding_manager = EmbeddingManager()
        self.llm_interface = LLMInterface(api_key, model)
        self.query_planner = QueryPlanner()
        self.reasoning_engine = ReasoningEngine()

        # State management
        self.documents_loaded = False
        self.total_chunks = 0
        self.processing_stats = {}

    def load_documents(self, uploaded_files) -> bool:
        """
        Process and load PDF documents into the system

        This is the setup phase of RAG:
        1. Extract text from PDFs
        2. Split into chunks
        3. Create embeddings
        4. Build search index

        Args:
            uploaded_files: List of Streamlit uploaded file objects

        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()

        try:
            st.write("🚀 Starting document processing pipeline...")

            # Step 1: Process PDFs and create chunks
            st.write("📄 **Step 1/3**: Extracting text from PDFs...")
            chunks = self.pdf_processor.process_pdf_files(uploaded_files)

            if not chunks:
                st.error("❌ No text could be extracted from the uploaded files")
                return False

            self.total_chunks = len(chunks)
            st.write(f"✅ Created {self.total_chunks} text chunks")

            # Step 2: Create embeddings
            st.write("🧠 **Step 2/3**: Creating vector embeddings...")
            success = self.embedding_manager.create_embeddings(chunks)

            if not success:
                st.error("❌ Failed to create embeddings")
                return False

            # Step 3: Finalize setup
            st.write("⚡ **Step 3/3**: Finalizing setup...")

            # Test LLM connection
            if not self.llm_interface.test_connection():
                st.warning("⚠️ LLM connection test failed, but documents are loaded")

            # Update state
            self.documents_loaded = True
            processing_time = time.time() - start_time

            # Store processing statistics
            self.processing_stats = {
                "total_files": len(uploaded_files),
                "total_chunks": self.total_chunks,
                "processing_time": processing_time,
                "embedding_stats": self.embedding_manager.get_embedding_stats()
            }

            st.success(f"🎉 **Documents loaded successfully!**")
            st.info(f"📊 Processed {len(uploaded_files)} files → {self.total_chunks} chunks in {processing_time:.2f}s")

            return True

        except Exception as e:
            st.error(f"❌ Error during document loading: {str(e)}")
            return False

    def query(self, user_query: str) -> RAGResult:
        """
        Process a user query using the agentic RAG pipeline

        This is where all the magic happens:
        1. Analyze query and create plan (Agentic)
        2. Retrieve relevant information (RAG)
        3. Generate response with reasoning (Agentic + RAG)

        Args:
            user_query: The user's question or request

        Returns:
            RAGResult containing answer and full reasoning process
        """
        if not self.documents_loaded:
            return RAGResult(
                answer="❌ No documents loaded. Please upload PDF files first.",
                query_plan=None,
                reasoning_steps=[],
                retrieved_chunks=[],
                response_metadata={},
                processing_time=0.0
            )

        start_time = time.time()

        try:
            st.write("🤔 **Analyzing your query...**")

            # Step 1: Query Analysis and Planning (AGENTIC)
            query_plan = self.query_planner.analyze_query(user_query)

            st.write(f"🎯 **Query Type Identified**: `{query_plan.query_type}`")
            st.write(f"📋 **Planned Approach**: {len(query_plan.steps)} steps")

            # Step 2: Initialize Reasoning Engine (AGENTIC)
            reasoning_session = self.reasoning_engine.start_reasoning_session(
                query=user_query,
                plan=query_plan
            )

            # Step 3: Information Retrieval (RAG)
            st.write("🔍 **Retrieving relevant information...**")

            retrieved_chunks = self.embedding_manager.similarity_search(
                query=user_query,
                top_k=query_plan.chunks_needed,
                threshold=query_plan.similarity_threshold
            )

            if not retrieved_chunks:
                return RAGResult(
                    answer="❌ I couldn't find relevant information in the uploaded documents to answer your query. Please try rephrasing your question or check if the documents contain the information you're looking for.",
                    query_plan=query_plan,
                    reasoning_steps=reasoning_session.steps,
                    retrieved_chunks=[],
                    response_metadata={"retrieval_failed": True},
                    processing_time=time.time() - start_time
                )

            # Update reasoning with retrieval results
            reasoning_session.add_step(
                f"Retrieved {len(retrieved_chunks)} relevant chunks with average similarity: {np.mean([score for _, score in retrieved_chunks]):.3f}"
            )

            st.write(f"✅ Found {len(retrieved_chunks)} relevant chunks")

            # Step 4: Response Generation (RAG + AGENTIC)
            st.write("✍️ **Generating comprehensive answer...**")

            response = self._generate_agentic_response(
                user_query=user_query,
                query_plan=query_plan,
                retrieved_chunks=retrieved_chunks,
                reasoning_session=reasoning_session
            )

            # Finalize reasoning
            reasoning_session.complete()

            processing_time = time.time() - start_time

            # Prepare metadata
            response_metadata = {
                "model_used": self.model,
                "chunks_retrieved": len(retrieved_chunks),
                "avg_similarity": float(np.mean([score for _, score in retrieved_chunks])),
                "processing_time": processing_time,
                "tokens_estimated": self.llm_interface.estimate_tokens(response.content)
            }

            st.success(f"✅ **Response generated** in {processing_time:.2f}s")

            return RAGResult(
                answer=response.content,
                query_plan=query_plan,
                reasoning_steps=reasoning_session.steps,
                retrieved_chunks=retrieved_chunks,
                response_metadata=response_metadata,
                processing_time=processing_time
            )

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            return RAGResult(
                answer=f"I encountered an error while processing your query: {str(e)}",
                query_plan=None,
                reasoning_steps=[],
                retrieved_chunks=[],
                response_metadata={"error": str(e)},
                processing_time=time.time() - start_time
            )

    def _generate_agentic_response(
            self,
            user_query: str,
            query_plan: QueryPlan,
            retrieved_chunks: List[Tuple[DocumentChunk, float]],
            reasoning_session
    ) -> LLMResponse:
        """
        Generate response using agentic reasoning and retrieved context

        Args:
            user_query: Original user question
            query_plan: Plan created by query planner
            retrieved_chunks: Retrieved document chunks with similarity scores
            reasoning_session: Current reasoning session

        Returns:
            LLMResponse with generated answer
        """

        # Prepare context from retrieved chunks
        context = self.llm_interface.format_context_for_llm(retrieved_chunks)

        # Create query-type specific system prompt
        system_prompt = self._create_system_prompt(query_plan.query_type)

        # Add reasoning context to system prompt
        reasoning_context = f"""
Current reasoning steps completed:
{chr(10).join([f"- {step.description}" for step in reasoning_session.steps])}

Planned remaining steps:
{chr(10).join([f"- {step}" for step in query_plan.steps[len(reasoning_session.steps):]])}
"""

        system_prompt += f"\n\nReasoning Context:\n{reasoning_context}"

        # Create messages for LLM
        messages = self.llm_interface.create_chat_messages(
            system_prompt=system_prompt,
            user_query=user_query,
            context=context
        )

        # Generate response with appropriate parameters for query type
        config = AgentConfig.QUERY_TYPE_THRESHOLDS.get(query_plan.query_type, {})
        max_tokens = config.get("max_tokens", 1000)

        response = self.llm_interface.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )

        # Update reasoning with generation step
        reasoning_session.add_step(f"Generated {len(response.content)} character response using {self.model}")

        return response

    def _create_system_prompt(self, query_type: str) -> str:
        """
        Create specialized system prompts for different query types

        Args:
            query_type: Type of query (summary, search, comparison, analysis)

        Returns:
            Tailored system prompt string
        """

        base_prompt = """You are an expert document analyst with advanced reasoning capabilities. Your task is to provide accurate, comprehensive, and well-structured responses based on the provided document context."""

        type_specific_prompts = {
            "summary": """
**Specialization**: Document Summarization
- Create comprehensive summaries that capture main points, key insights, and important details
- Structure your summary with clear sections and logical flow
- Highlight the most important information first
- Include specific examples and data points when available
""",

            "search": """
**Specialization**: Precise Question Answering  
- Provide direct, accurate answers to specific questions
- Quote relevant information from the source material
- If information is incomplete, clearly state what's missing
- Support answers with specific references to the source material
""",

            "comparison": """
**Specialization**: Comparative Analysis
- Identify key similarities and differences between entities/concepts
- Present comparisons in a clear, structured format
- Use tables or bullet points for clarity when appropriate
- Provide balanced analysis covering multiple perspectives
""",

            "analysis": """
**Specialization**: Deep Analysis
- Identify patterns, relationships, and underlying insights
- Provide thorough analysis with supporting evidence
- Draw meaningful conclusions based on the information
- Consider implications and broader context
"""
        }

        specific_prompt = type_specific_prompts.get(query_type, type_specific_prompts["search"])

        return base_prompt + specific_prompt + """

**Important Guidelines**:
- Base your response primarily on the provided context
- If context is insufficient, clearly state this limitation
- Use specific quotes and references to support your points
- Maintain objectivity and accuracy
- Structure your response for maximum clarity and usefulness
"""

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and statistics

        Returns:
            Dictionary with system status information
        """
        status = {
            "documents_loaded": self.documents_loaded,
            "total_chunks": self.total_chunks,
            "model_used": self.model,
            "embedding_model": self.embedding_manager.model_name
        }

        if self.documents_loaded:
            status.update(self.processing_stats)
            status["embedding_stats"] = self.embedding_manager.get_embedding_stats()

        return status

    def clear_documents(self):
        """
        Clear all loaded documents and reset the system
        """
        self.embedding_manager = EmbeddingManager()  # Reset embeddings
        self.documents_loaded = False
        self.total_chunks = 0
        self.processing_stats = {}

        st.info("🗑️ All documents cleared from memory")

# Import numpy for calculations
import numpy as np
