import time
import os
os.environ['HF_HOME'] = "D:/RFP_Finetuning/hf_cache"


from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
import fitz
import subprocess
from contextlib import contextmanager
import gc
import os
from PIL import Image
import io
import sys


load_dotenv()


if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    available_memory = total_memory - allocated_memory
    print(f"Total GPU memory: {total_memory:.1f} GB")
    print(f"Available GPU memory: {available_memory:.1f} GB")




class TransformerOCR:

    def __init__(self):

        try:
            if os.environ['VISION_MODEL']:
                raise ValueError(f"VISION MODEL key not set in environment ...")
            MODEL = os.environ['VISION_MODEL']
            # Load the model and processor
            self.processor = AutoProcessor.from_pretrained(
                MODEL,
                trust_remote_code=True
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL,
                torch_dtype=torch.float16,  # Reduce memory usage
                device_map="auto",         # Automatically distribute across available devices
                # low_cpu_mem_usage=True,   # Reduce CPU memory usage during loading
                trust_remote_code=True
            )
        except Exception as e :
            print(f"Error Initializingn Transformer OCR :- \n" , e)




    def extract_text_from_local_image(self , image_path, prompt="Please extract and return all the text visible in this image and covert to English language if the input is in other language."):
        """
        Extract text from a local image file using the vision-language model.
        
        Args:
            image_path (str): Path to the local image file
            prompt (str): Text prompt to instruct the model what to do with the image
        
        Returns:
            str: Extracted text from the image
        """
        # Verify that the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            # Open and load the local image
            image = Image.open(image_path)
            # Create the message structure with local image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},  # Use local image object instead of URL
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            # Process the input
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # Generate the response
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=1024,  # Increased for potentially longer text extraction
                do_sample=False,     # Deterministic output for text extraction
            )
            # Decode and return the generated text
            generated_text = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return generated_text.strip()
        except Exception as e :
            print(f"Error in performing OCR via transformer:- \n" , e)



class OllamaQwenOCR:
    def __init__(self , model = "qwen3-vl:4b"):
        try:

            self.llm = ChatOllama(
                model=model,  # or "qwen2.5-vl:7b" for better accuracy
                temperature=0.3,
                max_tokens=1024,
                # device_map = "auto"
            )
        except Exception as e :
            print(f"Error Initializing Qwen OCR :- \n" , e)

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string (Ollama expects this format)"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def ocr_with_qwen_vl(self ,image_path: str, prompt: str = None) -> str:
        try:
            # Default high-quality OCR prompt
            if prompt is None:
                prompt = (
                    "Perform high-accuracy OCR on this image. "
                    "and extract all visible text exactly as it appears. "
                    "Preserve formatting, line breaks, bullet points, and tables if present. "
                    "Return only the extracted text that too strictly in English language if language other than English is used then conver to English."
                )
            # Encode image
            base64_image = self.encode_image_to_base64(image_path)
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            )
            # Get response
            print("Processing image with Qwen-VL...")
            response = self.llm.invoke([message])
            return response.content
        except Exception as e :
            print(f"Error Performing OCR using Qwen via Ollama :- \n " , e)
    

class OllamaDeepSeekOCR:
    def __init__(self):
        self.memory_threshold = 0.85  # Stop if VRAM usage exceeds 85%
        
    def get_vram_usage(self):
        """Get current VRAM usage percentage"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", 
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(','))
                return used / total
        except:
            pass
        return 1.0  # Assume full usage if we can't determine
    
    @contextmanager
    def managed_ocr_session(self):
        """Context manager that ensures complete cleanup after each page"""
        try:
            yield
        finally:
            self.force_complete_cleanup()
    
    def force_complete_cleanup(self):
        """More aggressive cleanup than current approach"""
        print("Performing complete cleanup...")
        
        # Multiple sequential cleanup steps
        for i in range(3):  # Try multiple times
            subprocess.run(["ollama", "stop", "deepseek-ocr"], 
                         capture_output=True, timeout=15)
            time.sleep(1)
            
            # Try GPU reset, but don't fail if it doesn't work
            try:
                subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"], 
                             capture_output=True, timeout=10)
                time.sleep(1)
            except subprocess.TimeoutExpired:
                pass
        
        # Force Python garbage collection
        gc.collect()
        
        # Additional system-level cleanup
        try:
            subprocess.run(["nvidia-smi", "--gpu-reset", "-i", "0"], 
                         capture_output=True, timeout=5)
        except:
            pass
    
    def run_ocr_with_memory_management(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        all_text = []
        
        failed_pages = []
        
        for page_num in range(total_pages):
            # Check memory state before processing
            vram_usage = self.get_vram_usage()
            if vram_usage > self.memory_threshold:
                print(f"VRAM usage too high ({vram_usage:.1%}), skipping page {page_num + 1}")
                failed_pages.append(page_num + 1)
                self.force_complete_cleanup()
                continue
            
            print(f"\nProcessing page {page_num + 1}/{total_pages} (VRAM: {vram_usage:.1%})")
            
            temp_img_path = None
            try:
                # Render page
                page = doc[page_num]
                zoom = 2.5
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                
                
                temp_img_path = f"temp_ocr_page_{page_num}_{int(time.time()*1000000)}.png"
                img.save(temp_img_path, "PNG")

                try:
                    # Use Popen with communicate to ensure complete output capture
                    command = [
                        "ollama", "run", "deepseek-ocr",
                        f"{temp_img_path}\nExtract the text in the image"
                    ]
                    
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8"
                    )
                    
                    stdout, stderr = process.communicate(timeout=30)
                    
                    if process.returncode != 0:
                        extracted_text = f"(OCR failed: {stderr.strip()})"
                    else:
                        # Extract text from complete output
                        lines = []
                        in_response = False
                        for line in stdout.splitlines():
                            line = line.strip()
                            if line and "Added image" in line:
                                in_response = True  # Start capturing after image is added
                                continue
                            if in_response and line and not line.startswith(">>>"):
                                lines.append(line)
                        
                        extracted_text = "\n".join(lines).strip()

                        print(f"Extracted text :-" , extracted_text)
                    
                    all_text.append(f"\n--- Page {page_num + 1} ---\n{extracted_text}\n")

                except subprocess.TimeoutExpired:
                    process.kill()
                    all_text.append(f"\n--- Page {page_num + 1} ---\n(OCR timed out)\n")
            except Exception as e:
                all_text.append(f"\n--- Page {page_num + 1} ---\n(OCR error: {e})\n")
            finally:
                # Always cleanup
                if temp_img_path and os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                
                # Clean up after every page, regardless of success
                self.force_complete_cleanup()
                
                # Longer pause between pages to allow memory to fully settle
                time.sleep(1)
        
        if failed_pages:
            print(f"Warning: {len(failed_pages)} pages were skipped due to high memory usage: {failed_pages}")
        
        return "".join(all_text)

    # Additional strategy: Process in batches with complete restarts
    def run_ocr_with_complete_restarts(self, pdf_path: str, batch_size: int = 3):
        """Process the document in small batches with complete system restarts"""
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        all_text = []
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            print(f"Processing batch {batch_start//batch_size + 1}: pages {batch_start + 1} to {batch_end}")
            
            batch_text = []
            for page_num in range(batch_start, batch_end):
                # Process single page as before, but with the improved cleanup
                # ... (use the improved single-page processing from above)
                pass
            
            all_text.extend(batch_text)
            
            # After each batch, perform complete cleanup and optionally restart ollama
            subprocess.run(["ollama", "stop", "deepseek-ocr"], capture_output=True)
            time.sleep(2)  # Allow complete memory stabilization
            
            # Optional: completely restart the ollama service between batches
            subprocess.run(["net", "stop", "ollama"], capture_output=True)
            time.sleep(2)
            subprocess.run(["net", "start", "ollama"], capture_output=True)



# Example usage
if __name__ == "__main__":
    # Specify the path to your local image file
    image_path = "arabic.jpg"

    try:
        start_time = time.perf_counter()
        # Extract text from the local image
        extracted_text = extract_text_from_local_image(image_path)
        end_time = time.perf_counter()


        print(f"Text extraction time :-" , end_time - start_time)
        
        # Print the extracted text
        print("Extracted Text:")
        print("=" * 50)
        print(extracted_text)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred during text extraction: {e}")

    

    