import os
import tempfile
import subprocess
import requests
from urllib.parse import urlparse
from pathlib import Path

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import fitz
from PIL import Image
import io
import time

# Only import docx2pdf on Windows (it's Windows-only), fallback to libreoffice on Linux/macOS
try:
    from docx2pdf import convert  # Windows + macOS (with MS Word)
    DOCX2PDF_AVAILABLE = True
except ImportError:
    DOCX2PDF_AVAILABLE = False


class FileParser:
    def __init__(self, file_path_or_url: str, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_input = file_path_or_url.strip()
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
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
            soffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe" 
            result = subprocess.run([
                soffice_path, "--headless", "--convert-to", "pdf",
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
            raise ValueError("LibreOffice not installed. Install it or use Windows with MS Word.")

        raise ValueError("Failed to convert DOCX to PDF using all available methods")

    def parse_text(self) -> str:
        file_path = None
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
            documents = loader.load()  # Use .load() for simplicity, or lazy_load() if huge

            full_text = "\n".join([doc.page_content for doc in documents])
            print(f"Successfully extracted {len(full_text):,} characters from document")
            

            # This check if explicity for files where pdf pages are scanned images after removing the whitespaces and other characters if character count is zero then use OCR
            # Clean text for detection (remove whitespace, newlines, etc.)
            cleaned_text = re.sub(r"\s+", "", full_text.strip())
            alphanumeric_count = len(re.sub(r"[^a-zA-Z0-9]", "", cleaned_text))

            # If less than 50 meaningful characters → likely scanned → use OCR
            if alphanumeric_count < 50:
                print(f"Warning: Only {alphanumeric_count} alphanumeric chars found → Likely scanned PDF. Running DeepSeek-OCR...")
                ocr_text = self._run_deepseek_ocr(final_pdf_path)
                print(f"OCR completed: {len(ocr_text):,} characters extracted via DeepSeek-OCR")
                return ocr_text

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

    def _run_deepseek_ocr(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        all_text = []

        print(f"Starting SERIAL OCR on {total_pages} pages (VRAM fully cleared after every page)")

        for page_num in range(total_pages):
            start_time = time.time()

            # === 1. Render page to high-quality image ===
            page = doc[page_num]
            zoom = 2.5
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            temp_img_path = os.path.abspath(f"temp_ocr_page_{page_num}_{int(time.time()*1000000)}.png")
            img.save(temp_img_path, "PNG")

            try:
                # === 2. Run OCR exactly like your working CLI ===
                command = [
                    "ollama", "run", "deepseek-ocr",
                    f"{temp_img_path}\nExtract the text in the image"
                ]

                print(f"\n→ Processing page {page_num + 1}/{total_pages}...")

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=300
                )

                # === 3. Extract the real text ===
                if result.returncode != 0:
                    extracted_text = f"(OCR failed: {result.stderr.strip()})"
                else:
                    full_output = result.stdout
                    # Remove "Added image..." line and junk
                    lines = [line.strip() for line in full_output.splitlines()
                            if line.strip() and "Added image" not in line and not line.startswith(">>>")]
                    extracted_text = "\n".join(lines).strip()
                    if not extracted_text:
                        extracted_text = full_output.split("Added image", 1)[-1].strip()

                # === 4. FORCE UNLOAD MODEL + CLEAR GPU VRAM ===
                print("   Unloading model and clearing GPU VRAM...")
                subprocess.run(["ollama", "stop", "deepseek-ocr"], capture_output=True, timeout=30)

                # This is the nuclear option that ALWAYS frees VRAM on Windows
                gpu_reset = subprocess.run(
                    ["nvidia-smi", "--gpu-reset", "-i", "0"],
                    capture_output=True,
                    text=True
                )
                if gpu_reset.returncode == 0:
                    print("   GPU VRAM fully reset!")
                else:
                    print(f"   GPU reset warning: {gpu_reset.stderr.strip()}")

                # === 5. Append result ===
                page_header = f"\n--- Page {page_num + 1} ---"
                all_text.append(f"{page_header}\n{extracted_text}\n")

                elapsed = time.time() - start_time
                print(f"Completed page {page_num + 1}/{total_pages} in {elapsed:.1f}s")

            except subprocess.TimeoutExpired:
                all_text.append(f"\n--- Page {page_num + 1} ---\n(OCR timed out)\n")
                print("   Timeout → forcing unload...")
                subprocess.run(["ollama", "stop", "deepseek-ocr"], capture_output=True)
                subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"], capture_output=True)
            except Exception as e:
                all_text.append(f"\n--- Page {page_num + 1} ---\n(OCR error: {e})\n")
            finally:
                # Always delete temp image
                if os.path.exists(temp_img_path):
                    try: os.remove(temp_img_path)
                    except: pass

            # Optional: tiny pause to let system breathe
            time.sleep(1)

if __name__ == "__main__":

    
    obj = FileParser(file_path_or_url="https://compliancebotai.blob.core.windows.net/compliancebotdev/rfp/document/67b5be9c749273GORz1739964060.pdf")

    res = obj.parse_text()
    print("-"*100)
    print(len(res))
    print(res)
    print("-"*100)

