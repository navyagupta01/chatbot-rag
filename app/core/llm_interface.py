
import requests
import json
import time
import streamlit as st
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from config import AppConfig

@dataclass
class LLMResponse:
    """
    Structured response from LLM API
    Encapsulates both the generated text and metadata
    """
    content: str                    # Generated response text
    model: str                      # Model that generated the response
    tokens_used: Optional[int] = None      # Tokens consumed (if available)
    cost: Optional[float] = None           # API cost (if available)
    response_time: Optional[float] = None  # Time taken to generate

class LLMInterface:
    """
    Interface for communicating with Language Models via OpenRouter

    OpenRouter provides access to multiple LLMs through a single API,
    including free models perfect for demonstrations and learning.
    """

    def __init__(self, api_key: str, model: str = None):
        """
        Initialize LLM interface

        Args:
            api_key: OpenRouter API key
            model: Model identifier (defaults to config default)
        """
        self.api_key = api_key
        self.model = model or AppConfig.DEFAULT_MODEL
        self.base_url = AppConfig.OPENROUTER_BASE_URL

        # Request headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-username/agentic-rag-chatbot",  # Optional
            "X-Title": "Agentic RAG PDF Chatbot"  # Optional
        }

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            max_tokens: int = 1000,
            temperature: float = 0.7,
            stream: bool = False
    ) -> LLMResponse:
        """
        Generate a response using the configured LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
            stream: Whether to stream the response (not implemented yet)

        Returns:
            LLMResponse object containing generated text and metadata
        """

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,  # Nucleus sampling for quality
            "frequency_penalty": 0.1,  # Reduce repetition
            "presence_penalty": 0.1,   # Encourage diverse topics
        }

        # Record start time for performance tracking
        start_time = time.time()

        try:
            # Make API request
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60  # 60 second timeout
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            result = response.json()
            response_time = time.time() - start_time

            # Extract generated content
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']

                # Extract usage information if available
                tokens_used = None
                if 'usage' in result:
                    tokens_used = result['usage'].get('total_tokens')

                return LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used=tokens_used,
                    response_time=response_time
                )
            else:
                raise ValueError("No valid response received from LLM")

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Please try again.")
            return LLMResponse(
                content="I apologize, but the request timed out. Please try again with a shorter query.",
                model=self.model
            )

        except requests.exceptions.HTTPError as e:
            error_msg = self._handle_http_error(e, response)
            st.error(f"🔥 API Error: {error_msg}")
            return LLMResponse(
                content=f"I encountered an API error: {error_msg}",
                model=self.model
            )

        except requests.exceptions.RequestException as e:
            st.error(f"🌐 Network error: {str(e)}")
            return LLMResponse(
                content="I'm having trouble connecting to the language model. Please check your internet connection.",
                model=self.model
            )

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            st.error(f"📄 Response parsing error: {str(e)}")
            return LLMResponse(
                content="I received an unexpected response format. Please try again.",
                model=self.model
            )

    def _handle_http_error(self, error: requests.exceptions.HTTPError, response) -> str:
        """
        Handle and format HTTP errors from the API

        Args:
            error: The HTTP error that occurred
            response: The HTTP response object

        Returns:
            Formatted error message
        """
        status_code = response.status_code

        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', 'Unknown error')
        except:
            error_msg = f"HTTP {status_code} error"

        # Handle common error codes
        if status_code == 401:
            return "Invalid API key. Please check your OpenRouter API key."
        elif status_code == 402:
            return "Insufficient credits. Your API account may be out of credits."
        elif status_code == 429:
            return "Rate limit exceeded. Please wait a moment and try again."
        elif status_code == 500:
            return "Server error. The API service may be temporarily unavailable."
        else:
            return f"{error_msg} (Status: {status_code})"

    def create_chat_messages(
            self,
            system_prompt: str,
            user_query: str,
            context: str = None,
            conversation_history: List[Dict] = None
    ) -> List[Dict[str, str]]:
        """
        Create properly formatted messages for the LLM

        Args:
            system_prompt: Instructions for the LLM behavior
            user_query: The user's question or request
            context: Retrieved document context (for RAG)
            conversation_history: Previous messages in the conversation

        Returns:
            List of message dictionaries ready for API call
        """
        messages = []

        # Add system message
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)

        # Create user message with context
        user_content = user_query
        if context:
            user_content = f"""
Based on the following context from the document(s), please answer the question.

Context:
{context}

Question: {user_query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please indicate this clearly.
"""

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def test_connection(self) -> bool:
        """
        Test the API connection and authentication

        Returns:
            True if connection successful, False otherwise
        """
        test_messages = [
            {"role": "user", "content": "Hello, please respond with 'Connection successful!'"}
        ]

        try:
            response = self.generate_response(
                messages=test_messages,
                max_tokens=10,
                temperature=0.0
            )

            return "successful" in response.content.lower()

        except Exception as e:
            st.error(f"Connection test failed: {str(e)}")
            return False

    def get_available_models(self) -> List[str]:
        """
        Get list of available models from OpenRouter
        Note: This requires a separate API call to OpenRouter's models endpoint

        Returns:
            List of available model identifiers
        """
        try:
            models_url = "https://openrouter.ai/api/v1/models"
            response = requests.get(models_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            models_data = response.json()
            return [model['id'] for model in models_data.get('data', [])]

        except Exception as e:
            st.warning(f"Could not fetch available models: {str(e)}")
            return list(AppConfig.AVAILABLE_MODELS.values())

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count for text

        This is a rough approximation. Actual tokenization
        depends on the specific model being used.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token on average
        return len(text) // 4

    def format_context_for_llm(self, chunks_with_scores: List[tuple], max_context_length: int = 4000) -> str:
        """
        Format retrieved chunks into context for the LLM

        Args:
            chunks_with_scores: List of (DocumentChunk, similarity_score) tuples
            max_context_length: Maximum characters in context

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, (chunk, score) in enumerate(chunks_with_scores):
            # Format chunk with metadata
            chunk_text = f"[Source {i+1}] (Page {chunk.page_number}, Relevance: {score:.3f}, File: {chunk.source_file}):\n{chunk.text}\n"

            # Check if adding this chunk would exceed limit
            if current_length + len(chunk_text) > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)
