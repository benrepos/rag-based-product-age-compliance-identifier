"""
PDF Reader Module for UK Law Document Ingestion
Enhanced extraction with pdfplumber and OCR fallback
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import pdfplumber


class LegalDocumentReader:
    """Reads and extracts text from legal document PDFs"""
    
    def __init__(self, input_dir="raw_law_docs", output_dir="raw_laws_txt"):
        """
        Initialize the PDF reader
        
        Args:
            input_dir: Directory containing PDF files (relative to this file)
            output_dir: Directory to save extracted text files
        """
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / input_dir
        self.output_dir = self.base_dir / output_dir
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_with_pypdf(self, pdf_path: Path) -> str:
        """Baseline text extraction using pypdf"""
        reader = PdfReader(pdf_path)
        text_pages = []
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            text_pages.append(page_text)
        return "\n".join(text_pages)

    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Layout-aware text extraction using pdfplumber"""
        texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Use a small tolerance to avoid word smashing
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                texts.append(page_text)
        return "\n".join(texts)

    def _maybe_ocr_pdf(self, pdf_path: Path) -> Optional[Path]:
        """
        If ocrmypdf is available, run OCR to produce a searchable PDF copy and return its path.
        Returns None if OCR is not available or fails.
        """
        ocrmypdf_bin = shutil.which("ocrmypdf")
        if not ocrmypdf_bin:
            return None
        tmp_dir = Path(tempfile.mkdtemp(prefix="ocr_"))
        ocred_pdf = tmp_dir / f"{pdf_path.stem}_ocr.pdf"
        try:
            subprocess.run(
                [ocrmypdf_bin, "--skip-text", "--quiet", str(pdf_path), str(ocred_pdf)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ocred_pdf.exists():
                return ocred_pdf
        except Exception:
            return None
        return None

    def _text_quality_score(self, text: str, num_pages: int) -> float:
        """Heuristic quality: words per page after basic cleaning"""
        if num_pages <= 0:
            return 0.0
        words = re.findall(r"\w+", text)
        return len(words) / float(num_pages)

    def _normalize_text(self, text: str) -> str:
        """Light normalization to fix common leaflet/table artifacts."""
        s = text
        # Insert space between lowercase→Uppercase transitions (overTobacco → over Tobacco)
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        # Replace bullet chars with newline-hyphen
        s = s.replace("•", "\n- ")
        # Collapse excessive whitespace
        s = re.sub(r"[\t\r\f]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        # Ensure space after punctuation where missing
        s = re.sub(r"([.,;:])(\S)", r"\1 \2", s)
        return s

    def extract_text_from_pdf(self, pdf_path: Path):
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from all pages
        """
        try:
            print(f"Processing: {pdf_path.name}")
            # First try pdfplumber (better layout retention)
            plumber_text = self._extract_with_pdfplumber(pdf_path)
            pages_count = len(PdfReader(pdf_path).pages)
            quality = self._text_quality_score(plumber_text, pages_count)
            # If text looks sparse or obviously broken, try OCR then re-extract
            if quality < 50:  # heuristic: <50 words per page
                ocred = self._maybe_ocr_pdf(pdf_path)
                if ocred:
                    plumber_text = self._extract_with_pdfplumber(ocred)
                    quality = self._text_quality_score(plumber_text, pages_count)
            # Fallback to pypdf if plumber yields almost nothing
            if quality < 10:
                pypdf_text = self._extract_with_pypdf(pdf_path)
                if len(pypdf_text) > len(plumber_text):
                    plumber_text = pypdf_text

            normalized = self._normalize_text(plumber_text)
            print(f"✓ Extracted ~{len(normalized)} chars | pages={pages_count} | quality={quality:.1f} wpp")
            return normalized
        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}")
            return None
    
    def save_text(self, text, output_filename):
        """
        Save extracted text to a file
        
        Args:
            text: The text content to save
            output_filename: Name of the output file
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Saved to: {output_path}")
    
    def process_all_pdfs(self):
        """
        Process all PDF files in the input directory
        
        Returns:
            dict: Mapping of PDF filenames to output text filenames
        """
        # Collect PDFs case-insensitively (handles .pdf, .PDF, etc.)
        pdf_files = [p for p in self.input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            return {}
        
        print(f"\n{'='*60}")
        print(f"Found {len(pdf_files)} PDF file(s) to process")
        print(f"{'='*60}\n")
        
        processed_files = {}
        
        for pdf_path in pdf_files:
            print(f"\n--- Processing: {pdf_path.name} ---")
            # Skip if existing TXT is newer than the PDF (idempotent reruns)
            output_filename = pdf_path.stem + ".txt"
            txt_out = self.output_dir / output_filename
            if txt_out.exists() and txt_out.stat().st_mtime >= pdf_path.stat().st_mtime:
                print(f"↷ Skipping (up-to-date): {txt_out.name}")
                processed_files[pdf_path.name] = output_filename
                continue

            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                # Save extracted text
                self.save_text(text, output_filename)
                
                processed_files[pdf_path.name] = output_filename
            
            print()
        
        print(f"{'='*60}")
        print(f"Processing complete! {len(processed_files)} file(s) extracted")
        print(f"{'='*60}\n")
        
        return processed_files
    
    def process_single_pdf(self, pdf_filename):
        """
        Process a single specific PDF file
        
        Args:
            pdf_filename: Name of the PDF file to process
            
        Returns:
            str: Path to the output text file, or None if failed
        """
        pdf_path = self.input_dir / pdf_filename
        
        if not pdf_path.exists():
            print(f"Error: File not found: {pdf_path}")
            return None
        
        print(f"\n--- Processing: {pdf_filename} ---")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if text:
            # Create output filename
            output_filename = pdf_path.stem + ".txt"
            
            # Save extracted text
            self.save_text(text, output_filename)
            
            return str(self.output_dir / output_filename)
        
        return None


def main():
    """Main function to run the PDF reader"""
    reader = LegalDocumentReader()
    reader.process_all_pdfs()


if __name__ == "__main__":
    main()
