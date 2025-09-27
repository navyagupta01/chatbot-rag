"""
🔍 Vector Search Utilities
===========================

Helper functions for vector similarity search operations.
Additional utilities for FAISS index management and optimization.
"""

import numpy as np
import faiss
from typing import List, Tuple, Optional
import streamlit as st

class VectorSearchUtils:
    """
    Utility class for vector search operations
    Provides additional functionality for FAISS index management
    """

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity

        Args:
            vectors: Input vectors to normalize

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    @staticmethod
    def calculate_similarity_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity scores between query and document vectors

        Args:
            query_vector: Query vector (1D)
            document_vectors: Document vectors (2D)

        Returns:
            Similarity scores
        """
        # Normalize vectors
        query_norm = VectorSearchUtils.normalize_vectors(query_vector.reshape(1, -1))
        doc_norm = VectorSearchUtils.normalize_vectors(document_vectors)

        # Calculate cosine similarity
        similarities = np.dot(doc_norm, query_norm.T).flatten()
        return similarities

    @staticmethod
    def filter_by_threshold(results: List[Tuple], threshold: float) -> List[Tuple]:
        """
        Filter search results by similarity threshold

        Args:
            results: List of (item, score) tuples
            threshold: Minimum similarity score

        Returns:
            Filtered results
        """
        return [(item, score) for item, score in results if score >= threshold]

    @staticmethod
    def diversify_results(results: List[Tuple], max_results: int = 5, diversity_threshold: float = 0.8) -> List[Tuple]:
        """
        Diversify search results to avoid redundancy

        Args:
            results: List of (chunk, score) tuples
            max_results: Maximum number of results to return
            diversity_threshold: Minimum diversity threshold

        Returns:
            Diversified results
        """
        if not results:
            return results

        diversified = [results[0]]  # Always include the top result

        for chunk, score in results[1:]:
            if len(diversified) >= max_results:
                break

            # Check diversity against already selected chunks
            is_diverse = True
            for selected_chunk, _ in diversified:
                # Simple text-based diversity check
                overlap = VectorSearchUtils._calculate_text_overlap(chunk.text, selected_chunk.text)
                if overlap > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                diversified.append((chunk, score))

        return diversified

    @staticmethod
    def _calculate_text_overlap(text1: str, text2: str) -> float:
        """
        Calculate text overlap between two strings

        Args:
            text1: First text
            text2: Second text

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def optimize_faiss_index(index: faiss.Index, vectors: np.ndarray) -> faiss.Index:
        """
        Optimize FAISS index for better performance

        Args:
            index: Current FAISS index
            vectors: Vector data

        Returns:
            Optimized FAISS index
        """
        # For small datasets, flat index is usually best
        if len(vectors) < 10000:
            return index

        # For larger datasets, consider using IVF index
        try:
            ncentroids = min(int(np.sqrt(len(vectors))), 256)
            quantizer = faiss.IndexFlatIP(vectors.shape[1])
            optimized_index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], ncentroids)
            optimized_index.train(vectors.astype('float32'))
            optimized_index.add(vectors.astype('float32'))
            return optimized_index
        except:
            # Fall back to original index if optimization fails
            return index
