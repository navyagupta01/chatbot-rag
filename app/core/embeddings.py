import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional
from .pdf_processor import DocumentChunk  # Assuming DocumentChunk is from your pdf_processor
from config import AppConfig


class EmbeddingManager:
    """
    Manages text embeddings and vector search operations WITH DEBUGGING
    """

    def __init__(self, model_name: str = None):
        """Initialize the embedding manager"""
        self.model_name = model_name or AppConfig.EMBEDDING_MODEL
        self.embedding_dimension = AppConfig.EMBEDDING_DIMENSION
        # Load the model
        st.write(f"🧠 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        st.success(f"✅ Model loaded: {self.model_name}")

        # FAISS index for fast similarity search
        self.faiss_index = None
        self.document_chunks = []

    def create_embeddings(self, chunks: List[DocumentChunk]) -> bool:
        """Create embeddings with debugging output"""

        if not chunks:
            st.warning("⚠️ DEBUG: No chunks provided for embedding creation")
            return False

        try:
            # Extract text from all chunks
            texts = [chunk.text for chunk in chunks]

            st.write(f"🔄 DEBUG: Creating embeddings for {len(texts)} chunks...")
            st.write(f"📝 DEBUG: First chunk sample: {texts[0][:100]}...")

            # Create embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            st.write(f"✅ DEBUG: Embeddings shape: {embeddings.shape}")

            # Store embeddings in chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding

            # Build FAISS index
            self._build_faiss_index(embeddings, chunks)

            st.success(f"🎉 Successfully created {len(embeddings)} embeddings!")
            return True

        except Exception as e:
            st.error(f"❌ DEBUG: Error creating embeddings: {str(e)}")
            return False

    def _build_faiss_index(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """Build FAISS index with debugging"""

        try:
            # Normalize embeddings for cosine similarity
            embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            st.write(f"📊 DEBUG: Normalized embeddings shape: {embeddings_normalized.shape}")

            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)

            # Add embeddings to index
            self.faiss_index.add(embeddings_normalized.astype('float32'))

            # Store chunks
            self.document_chunks = chunks

            st.write(f"🏗️ DEBUG: FAISS index built with {self.faiss_index.ntotal} vectors")

        except Exception as e:
            st.error(f"❌ DEBUG: Error building FAISS index: {str(e)}")
            raise

    def similarity_search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform similarity search with extensive debugging
        """

        if self.faiss_index is None or not self.document_chunks:
            st.warning("⚠️ DEBUG: No embeddings available for search")
            return []

        try:
            st.write(f"🔍 DEBUG: Searching for query: '{query}'")
            st.write(f"📊 DEBUG: Using threshold: {threshold}, top_k: {top_k}")

            # Create embedding for query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            st.write(f"🎯 DEBUG: Query embedding shape: {query_embedding.shape}")

            # Normalize query embedding
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)

            # Search in FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding_normalized.astype('float32'),
                min(top_k, len(self.document_chunks))
            )

            st.write(f"📈 DEBUG: Raw similarity scores: {similarities[0]}")
            st.write(f"🎲 DEBUG: Indices: {indices[0]}")

            # Prepare results with debugging
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                st.write(f"📋 DEBUG: Chunk {i}: idx={idx}, score={similarity:.3f}")

                if idx < len(self.document_chunks):
                    chunk = self.document_chunks[idx]

                    if similarity >= threshold:
                        results.append((chunk, float(similarity)))
                        st.write(f"✅ DEBUG: INCLUDED - Score {similarity:.3f} >= threshold {threshold}")
                        st.write(f"📄 DEBUG: Chunk preview: {chunk.text[:100]}...")
                    else:
                        st.write(f"❌ DEBUG: REJECTED - Score {similarity:.3f} < threshold {threshold}")

            if not results:
                st.error(f"🚫 DEBUG: NO CHUNKS PASSED THRESHOLD")
                st.error(f"💡 DEBUG: Best score was {similarities[0][0]:.3f}, threshold was {threshold}")
                st.error(f"🔧 DEBUG: Consider lowering threshold or check document content")

                # Emergency fallback - return top chunk regardless of threshold
                if len(similarities[0]) > 0:
                    best_idx = indices[0][0]
                    if best_idx < len(self.document_chunks):
                        st.warning(f"🆘 DEBUG: Emergency fallback - using best chunk with score {similarities[0][0]:.3f}")
                    results.append((self.document_chunks[best_idx], float(similarities[0][0])))

            else:
                st.success(f"🎯 DEBUG: Found {len(results)} relevant chunks!")

            return results

        except Exception as e:
            st.error(f"❌ DEBUG: Error during similarity search: {str(e)}")
            return []

    def get_embedding_stats(self) -> dict:
        """Get statistics about current embeddings"""

        if not self.document_chunks:
            return {"total_chunks": 0, "total_files": 0}

        # Count unique files
        unique_files = set(chunk.source_file for chunk in self.document_chunks)

        # Calculate average chunk size
        avg_chunk_size = np.mean([chunk.char_count for chunk in self.document_chunks])

        return {
            "total_chunks": len(self.document_chunks),
            "total_files": len(unique_files),
            "avg_chunk_size": int(avg_chunk_size),
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name
        }
