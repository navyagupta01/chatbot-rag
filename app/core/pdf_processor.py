import PyPDF2
import re
import streamlit as st
from typing import Dict, List, Optional
from dataclasses import dataclass
from io import BytesIO

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a PDF document"""
    text: str
    page_number: int
    chunk_id: str
    source_file: str
    char_count: int
    embedding: Optional[object] = None

class PDFProcessor:
    """
    PDF Processing Engine with debugging
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """Initialize PDF processor"""
        self.chunk_size = chunk_size
        self.overlap = overlap
        st.write(f"📄 DEBUG: PDF Processor initialized - chunk_size: {chunk_size}, overlap: {overlap}")

    def process_pdf_files(self, uploaded_files) -> List[DocumentChunk]:
        """Process multiple PDF files with debugging"""

        all_chunks = []

        for uploaded_file in uploaded_files:
            st.write(f"📁 DEBUG: Processing file: {uploaded_file.name}")

            try:
                # Extract text from PDF
                pages_text = self._extract_text_from_pdf(uploaded_file)

                if not pages_text:
                    st.warning(f"⚠️ DEBUG: No text extracted from {uploaded_file.name}")
                    continue

                # Create chunks
                file_chunks = self._create_chunks(pages_text, uploaded_file.name)
                all_chunks.extend(file_chunks)

                st.success(f"✅ DEBUG: {uploaded_file.name}: {len(pages_text)} pages → {len(file_chunks)} chunks")

            except Exception as e:
                st.error(f"❌ DEBUG: Error processing {uploaded_file.name}: {str(e)}")
                continue

        st.write(f"🎯 DEBUG: Total chunks created: {len(all_chunks)}")
        return all_chunks

    def _extract_text_from_pdf(self, uploaded_file) -> Dict[int, str]:
        """Extract text from PDF with debugging"""

        try:
            # Read PDF
            pdf_bytes = BytesIO(uploaded_file.read())
            pdf_reader = PyPDF2.PdfReader(pdf_bytes)

            st.write(f"📖 DEBUG: PDF has {len(pdf_reader.pages)} pages")

            pages_text = {}

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    # Extract text
                    raw_text = page.extract_text()
                    cleaned_text = self._clean_text(raw_text)

                    st.write(f"📄 DEBUG: Page {page_num + 1}: {len(raw_text)} → {len(cleaned_text)} chars")

                    if cleaned_text.strip():
                        pages_text[page_num + 1] = cleaned_text
                        # Show first 100 chars as preview
                        preview = cleaned_text[:100].replace('\n', ' ')
                        st.write(f"📝 DEBUG: Page {page_num + 1} preview: {preview}...")
                    else:
                        st.warning(f"⚠️ DEBUG: Page {page_num + 1} is empty after cleaning")

                except Exception as e:
                    st.warning(f"⚠️ DEBUG: Error extracting page {page_num + 1}: {str(e)}")
                    continue

            st.write(f"📊 DEBUG: Successfully extracted {len(pages_text)} non-empty pages")
            return pages_text

        except Exception as e:
            st.error(f"❌ DEBUG: Failed to read PDF: {str(e)}")
            return {}

    def _clean_text(self, text: str) -> str:
        """Clean extracted text with debugging"""

        if not text:
            return ""

        original_length = len(text)

        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)

        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase spacing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # sentence spacing

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '--', text)

        cleaned_text = text.strip()

        # Debug output
        if original_length != len(cleaned_text):
            st.write(f"🧹 DEBUG: Text cleaned - {original_length} → {len(cleaned_text)} chars")

        return cleaned_text

    def _create_chunks(self, pages_text: Dict[int, str], filename: str) -> List[DocumentChunk]:
        """Create chunks with debugging"""

        chunks = []
        chunk_counter = 0

        st.write(f"✂️ DEBUG: Creating chunks from {len(pages_text)} pages")

        for page_num, page_text in pages_text.items():
            st.write(f"✂️ DEBUG: Chunking page {page_num} ({len(page_text)} chars)")

            # Split into sentences for better boundaries
            sentences = self._split_into_sentences(page_text)
            st.write(f"📝 DEBUG: Page {page_num} split into {len(sentences)} sentences")

            current_chunk = ""

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Check if adding sentence exceeds chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

                if len(potential_chunk) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk = DocumentChunk(
                        text=current_chunk.strip(),
                        page_number=page_num,
                        chunk_id=f"{filename}_chunk_{chunk_counter}",
                        source_file=filename,
                        char_count=len(current_chunk.strip())
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                    # Start new chunk with overlap
                    if len(current_chunk) > self.overlap:
                        overlap_text = current_chunk[-self.overlap:]
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = potential_chunk

            # Save last chunk of page
            if current_chunk.strip():
                chunk = DocumentChunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    chunk_id=f"{filename}_chunk_{chunk_counter}",
                    source_file=filename,
                    char_count=len(current_chunk.strip())
                )
                chunks.append(chunk)
                chunk_counter += 1

            st.write(f"✅ DEBUG: Page {page_num} processed - Total chunks so far: {len(chunks)}")

        # Show chunk statistics
        if chunks:
            avg_chunk_size = sum(c.char_count for c in chunks) / len(chunks)
            st.write(f"📊 DEBUG: Created {len(chunks)} chunks, avg size: {avg_chunk_size:.0f} chars")
            st.write(f"📄 DEBUG: First chunk sample: {chunks[0].text[:100]}...")

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting pattern
        sentence_pattern = r'(?<![A-Z][a-z])(?<![A-Z])(?<=\.|\!|\?)\s+'
        sentences = re.split(sentence_pattern, text)

        return [s.strip() for s in sentences if s.strip()]
