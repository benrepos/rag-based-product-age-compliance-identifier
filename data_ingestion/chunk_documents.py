"""
Document Chunking Module for UK Law Documents
Uses tiktoken for token-aware chunking (500-800 tokens per chunk)
"""

import json
import re
from pathlib import Path
import tiktoken


class LegalDocumentChunker:
    """Chunks legal documents into token-aware segments with metadata"""
    
    def __init__(self, 
                 input_dir="raw_laws_txt",
                 output_dir="../embeddings",
                 min_chunk_size=500,
                 max_chunk_size=800,
                 model="text-embedding-3-small"):
        """
        Initialize the document chunker
        
        Args:
            input_dir: Directory containing extracted text files
            output_dir: Directory to save chunked data
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            model: OpenAI model name (for tokenizer)
        """
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / input_dir
        self.output_dir = Path(self.base_dir / output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Initialize tokenizer (use cl100k_base for text-embedding-3-small)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        """Count tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def extract_sections(self, text, law_name):
        """
        Extract sections from legal text
        Attempts to identify section markers and split accordingly
        
        Args:
            text: Full legal text
            law_name: Name of the law document
            
        Returns:
            list: List of tuples (section_id, section_text)
        """
        sections = []
        
        # Common patterns for legal sections
        section_patterns = [
            r'\n(\d+)\s+([A-Z][^\n]{10,100})\n',  # "1 Title of section"
            r'\nSection\s+(\d+)\s*[:\-]?\s*([^\n]*)\n',  # "Section 1: Title"
            r'\n([A-Z]+\s+\d+)\s+([A-Z][^\n]{10,100})\n',  # "PART 1 Title"
        ]
        
        # Try to find section breaks
        all_matches = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                all_matches.extend([(m.start(), m.group(0), m.group(1)) for m in matches])
        
        if all_matches:
            # Sort by position
            all_matches.sort(key=lambda x: x[0])
            
            # Create sections based on matches
            for i, (pos, match_text, section_id) in enumerate(all_matches):
                start = pos
                end = all_matches[i + 1][0] if i + 1 < len(all_matches) else len(text)
                section_text = text[start:end].strip()
                
                if section_text:
                    sections.append((section_id, section_text))
        
        # If no sections found, treat the whole document as one section
        if not sections:
            sections = [("Full Document", text)]
        
        return sections
    
    def chunk_text(self, text, section_id, law_name):
        """
        Split text into token-aware chunks
        
        Args:
            text: Text to chunk
            section_id: Section identifier
            law_name: Name of the law
            
        Returns:
            list: List of chunk dictionaries with metadata
        """
        chunks = []
        
        # Split into paragraphs first (better semantic boundaries)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = []
        current_tokens = 0
        chunk_idx = 0
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            # If a single paragraph exceeds max size, split it by sentences
            if para_tokens > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk_metadata(
                        ' '.join(current_chunk), 
                        law_name, 
                        section_id, 
                        chunk_idx
                    ))
                    chunk_idx += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self.count_tokens(sent)
                    
                    if current_tokens + sent_tokens > self.max_chunk_size and current_chunk:
                        chunks.append(self._create_chunk_metadata(
                            ' '.join(current_chunk),
                            law_name,
                            section_id,
                            chunk_idx
                        ))
                        chunk_idx += 1
                        current_chunk = [sent]
                        current_tokens = sent_tokens
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens
            
            # Normal paragraph processing
            elif current_tokens + para_tokens > self.max_chunk_size:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(self._create_chunk_metadata(
                        ' '.join(current_chunk),
                        law_name,
                        section_id,
                        chunk_idx
                    ))
                    chunk_idx += 1
                
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        # Save final chunk
        # Always save the tail chunk if it's the only chunk for this section,
        # even if it is smaller than min_chunk_size. This prevents empty outputs
        # for documents with many short sections.
        if current_chunk and (current_tokens >= self.min_chunk_size or not chunks):
            chunks.append(self._create_chunk_metadata(
                ' '.join(current_chunk),
                law_name,
                section_id,
                chunk_idx
            ))
        
        return chunks
    
    def _create_chunk_metadata(self, text, law_name, section_id, chunk_idx):
        """
        Create a chunk dictionary with metadata
        
        Args:
            text: Chunk text
            law_name: Name of the law
            section_id: Section identifier
            chunk_idx: Index of chunk within section
            
        Returns:
            dict: Chunk with metadata
        """
        token_count = self.count_tokens(text)
        
        return {
            "chunk_id": f"{law_name}_{section_id}_{chunk_idx}",
            "law_name": law_name,
            "section": section_id,
            "chunk_index": chunk_idx,
            "chunk_text": text,
            "token_count": token_count,
            "url": f"https://www.legislation.gov.uk/ukpga/2003/17/section/{section_id}"  # Generic URL template
        }
    
    def _detect_doc_type(self, text: str) -> str:
        """Heuristic doc type: 'law' for Acts/Regs, 'guidance' for leaflets/tables"""
        # If there are many numbered sections or 'PART X' headings, treat as law
        law_hits = len(re.findall(r"\n(PART\s+\d+|Section\s+\d+|\n\d+\s+[A-Z])", text))
        if law_hits >= 10 or "An Act to" in text[:500]:
            return "law"
        return "guidance"

    def process_document(self, txt_file_path):
        """
        Process a single text file into chunks
        
        Args:
            txt_file_path: Path to the text file
            
        Returns:
            list: List of chunks with metadata
        """
        law_name = txt_file_path.stem  # Filename without extension
        
        print(f"\n--- Processing: {law_name} ---")
        
        # Read the text file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"Total characters: {len(text):,}")
        print(f"Total tokens: {self.count_tokens(text):,}")
        doc_type = self._detect_doc_type(text)
        print(f"Detected doc_type: {doc_type}")
        
        # Extract sections
        sections = self.extract_sections(text, law_name)
        print(f"Found {len(sections)} section(s)")
        
        # Chunk each section
        all_chunks = []
        for section_id, section_text in sections:
            # For guidance, temporarily reduce min_chunk threshold to capture short lines
            if doc_type == "guidance":
                original_min = self.min_chunk_size
                original_max = self.max_chunk_size
                try:
                    self.min_chunk_size = 80
                    self.max_chunk_size = max(300, self.max_chunk_size)
                    section_chunks = self.chunk_text(section_text, section_id, law_name)
                finally:
                    self.min_chunk_size = original_min
                    self.max_chunk_size = original_max
            else:
                section_chunks = self.chunk_text(section_text, section_id, law_name)
            all_chunks.extend(section_chunks)
        
        print(f"Created {len(all_chunks)} chunk(s)")
        if all_chunks:
            print(f"Token range: {min(c['token_count'] for c in all_chunks)}-{max(c['token_count'] for c in all_chunks)} tokens")
        else:
            print("Token range: n/a (no chunks produced); consider lowering min_chunk_size or merging strategy")
        
        return all_chunks
    
    def process_all_documents(self):
        """
        Process all text files in the input directory
        
        Returns:
            list: All chunks from all documents
        """
        txt_files = list(self.input_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"No text files found in {self.input_dir}")
            return []
        
        print(f"\n{'='*70}")
        print(f"📜 CHUNKING {len(txt_files)} LEGAL DOCUMENT(S)")
        print(f"{'='*70}")
        print(f"Target chunk size: {self.min_chunk_size}-{self.max_chunk_size} tokens")
        
        all_chunks = []
        
        for txt_file in txt_files:
            chunks = self.process_document(txt_file)
            all_chunks.extend(chunks)
        
        print(f"\n{'='*70}")
        print(f"✓ Total chunks created: {len(all_chunks)}")
        print(f"{'='*70}\n")
        
        # Save to JSON
        output_file = self.output_dir / "law_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved chunks to: {output_file}")
        print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")
        
        return all_chunks


def main():
    """Main function to run document chunking"""
    chunker = LegalDocumentChunker()
    chunks = chunker.process_all_documents()
    return chunks


if __name__ == "__main__":
    main()

