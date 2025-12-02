import os
import tempfile
import subprocess
import requests
from urllib.parse import urlparse
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import fitz
import io
import time

# Only import docx2pdf on Windows (it's Windows-only), fallback to libreoffice on Linux/macOS
try:
    from docx2pdf import convert  # Windows + macOS (with MS Word)
    DOCX2PDF_AVAILABLE = True
except ImportError:
    DOCX2PDF_AVAILABLE = False


class FileParser:

    def __init__(self):
        
        self.temp_file_to_cleanup = None  # Track file to delete if downloaded
    
    def classify_path_or_url(self, s: str) -> str:
        parsed = urlparse(s)
        if parsed.scheme in ("http", "https", "ftp") and parsed.netloc:
            return "url"
        if os.path.isabs(s) or os.path.isfile(s) or "." in Path(s).suffix:
            return "file_path"
        return "unknown"

    def download_file(self, url: str) -> str:
        """Download file from URL and return local temp path"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Guess extension from URL or Content-Type
            ext = ""
            if "." in url.split("/")[-1]:
                ext = "." + url.split("/")[-1].split(".")[-1].split("?")[0]
            elif response.headers.get("content-type"):
                ctype = response.headers["content-type"]
                if "pdf" in ctype:
                    ext = ".pdf"
                elif "msword" in ctype or "officedocument" in ctype:
                    ext = ".docx"

            fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="uploaded_")
            os.close(fd)  # We just need the path

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Downloaded file saved temporarily: {temp_path}")
            self.temp_file_to_cleanup = temp_path
            return temp_path

        except Exception as e:
            raise ValueError(f"Failed to download file from URL: {e}")

    def convert_docx_to_pdf(self, docx_path: str) -> str:
        """Convert DOCX to PDF using docx2pdf (Windows) or libreoffice (Linux/macOS)"""
        pdf_path = os.path.splitext(docx_path)[0] + ".pdf"

        if DOCX2PDF_AVAILABLE and os.name == "nt":  # Windows
            try:
                convert(docx_path, pdf_path)
                if os.path.exists(pdf_path):
                    return pdf_path
            except Exception as e:
                print(f"docx2pdf failed: {e}")

        # Fallback: use LibreOffice (works on Linux/macOS, and Windows if installed)
        try: 
            result = subprocess.run([
                "libreoffice", "--headless", "--convert-to", "pdf",
                "--outdir", os.path.dirname(docx_path), docx_path
            ], check=True, capture_output=True, timeout=60)

            if os.path.exists(pdf_path):
                return pdf_path
            else:
                # Sometimes libreoffice names output differently
                expected = os.path.join(os.path.dirname(docx_path), Path(docx_path).stem + ".pdf")
                if os.path.exists(expected):
                    return expected
        except subprocess.TimeoutExpired:
            raise ValueError("LibreOffice conversion timed out")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"LibreOffice conversion failed: {e}")
        except FileNotFoundError:
            raise ValueError(f"File not found :- {self.file_input}")
        

        raise ValueError("Failed to convert DOCX to PDF using all available methods")

    def parse_text(self , file_path_or_url : str ) -> str:

        self.file_input = file_path_or_url.strip()
        
        try:
            input_type = self.classify_path_or_url(self.file_input)

            if input_type == "unknown":
                raise ValueError("Input is neither a valid file path nor a URL")

            if input_type == "url":
                file_path = self.download_file(self.file_input)
            else:  # file_path
                if not os.path.isfile(self.file_input):
                    raise ValueError(f"File not found: {self.file_input}")
                file_path = self.file_input

            # Determine file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip(".")

            if ext not in ["pdf", "docx"]:
                raise ValueError("Only .pdf and .docx files are supported")

            # Convert DOCX to PDF if needed
            final_pdf_path = file_path
            if ext == "docx":
                print("Converting DOCX to PDF...")
                final_pdf_path = self.convert_docx_to_pdf(file_path)
                # If conversion created new file, mark original for cleanup only if it was downloaded
                if input_type == "url" and file_path != final_pdf_path:
                    # Keep track of both if needed, but usually only the original temp file
                    pass

            # Load text using PyMuPDF4LLMLoader (great for tables + text)
            print(f"Loading document with PyMuPDF4LLMLoader: {final_pdf_path}")
            loader = PyMuPDF4LLMLoader(file_path=final_pdf_path)
            documents = loader.lazy_load()

            full_text = "\n".join([doc.page_content for doc in documents])
            print(f"Successfully extracted {len(full_text):,} characters from document")
            

            # This check if explicity for files where pdf pages are scanned images after removing the whitespaces and other characters if character count is zero then use OCR
            # Clean text for detection (remove whitespace, newlines, etc.)
            cleaned_text = re.sub(r"\s+", "", full_text.strip())
            alphanumeric_count = len(re.sub(r"[^a-zA-Z0-9]", "", cleaned_text))

            # If less than 50 meaningful characters → likely scanned → use OCR
            if alphanumeric_count < 50:
                print(f"Warning: Very few text was extracted  → Likely scanned PDF.... skipping the document ...")
                return ""
                ###################### Instead of returning empty string we can fallback to OCR (currently not implemented since it takes lot of time ) given hardware limitation ##############
            return full_text
        
        except Exception as e:
            print(f"Error during parsing: {e}")
            raise  # Re-raise for caller to handle
        finally:
            # Cleanup: only delete temp files created by us
            if self.temp_file_to_cleanup and os.path.exists(self.temp_file_to_cleanup):
                try:
                    os.unlink(self.temp_file_to_cleanup)
                    print(f"Cleaned up temporary file: {self.temp_file_to_cleanup}")
                except:
                    pass
            self.temp_file_to_cleanup = None

    def split_text(self, text: str):
        """
        Split extracted text into appropriately sized chunks
        Returns a list of text chunks suitable for embedding, fine-tuning, or retrieval
        """
        if not text.strip():
            return []
        
        print(f"Splitting {len(text):,} characters into chunks of size {self.max_chunk_size}")
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        
        print(f"Created {len(chunks)} chunks")
        
        # Optional: Add metadata about chunk sizes
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = total_chunk_length / len(chunks) if chunks else 0
        
        print(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        return chunks





if __name__ == "__main__":

    
    obj = FileParser()

    res = obj.parse_text(file_path_or_url="document.pdf")
    print("-"*100)
    print(len(res))
    print(res)
    print("-"*100)

