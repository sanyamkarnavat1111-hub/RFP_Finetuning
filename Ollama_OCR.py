# ocr_with_qwen_vl.py

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
import time


def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 string (Ollama expects this format)"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ocr_with_qwen_vl(image_path: str, prompt: str = None) -> str:
    # Load the Qwen-VL model via Ollama
    llm = ChatOllama(
        model="qwen3-vl:4b",  # or "qwen2.5-vl:7b" for better accuracy
        temperature=0.3,
        max_tokens=1024,
        # device_map = "auto"
    )

    # Default high-quality OCR prompt
    if prompt is None:
        prompt = (
            "Perform high-accuracy OCR on this image. "
            "and extract all visible text exactly as it appears. "
            "Preserve formatting, line breaks, bullet points, and tables if present. "
            "Return only the extracted text that too strictly in English language if language other than English is used then conver to English."
        )

    # Encode image
    base64_image = encode_image_to_base64(image_path)

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
    response = llm.invoke([message])
    return response.content

# ============== USAGE EXAMPLE ==============
if __name__ == "__main__":
    # Replace with your image path
    image_path = "arabic.jpg"  # Put your image here (PNG, JPG, etc.)

    try:
        start_time = time.perf_counter()
        extracted_text = ocr_with_qwen_vl(image_path)
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(extracted_text)
        end_time = time.perf_counter()
        print(f"Total time required ... :- " , end_time-start_time)
    except Exception as e:
        print("Error:", e)